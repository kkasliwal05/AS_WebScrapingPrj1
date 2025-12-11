# app.py -- Simple query-only API: accept CSV + query JSON -> return response JSON
import os
import re
import json
import time
from io import BytesIO, StringIO  # <-- added StringIO
from typing import Optional

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional NLTK pieces used for sentence splitting
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# NEW: WebSocket (Socket.IO)
from flask_socketio import SocketIO, emit

# minimal nltk assets required (download if missing)
_nltk_needed = ['punkt', 'wordnet', 'omw-1.4']
for r in _nltk_needed:
    try:
        nltk.data.find(r)
    except Exception:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

# ---------- Config ----------
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SIMILARITY_THRESHOLD = 0.20
TOP_K_DEFAULT = 3

# ---------- Helpers ----------
IRRELEVANT_KEYWORDS = [
    "gym", "dumbbell", "exercise", "workout", "recipe", "food", "cooking",
    "weather", "news", "politics", "relationship", "doctor", "medicine",
    "math", "code", "python", "java", "cpp"
]
_irrelevant_regex = re.compile(r'\b(' + r'|'.join(re.escape(w) for w in IRRELEVANT_KEYWORDS) + r')\b', flags=re.I)

def is_irrelevant(query: str) -> bool:
    if not query or str(query).strip() == "":
        return True
    return bool(_irrelevant_regex.search(query))

def safe_jsonify(obj):
    """Convert numpy / pandas types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {safe_jsonify(k): safe_jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_jsonify(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # handle pandas NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

# ---------- Core retrieval utilities ----------
def choose_text_field(df: pd.DataFrame) -> str:
    """Pick the best field to index for retrieval (prefer indexed_text_lemma, then search_content, else first text-like)."""
    if 'indexed_text_lemma' in df.columns and df['indexed_text_lemma'].astype(str).str.strip().any():
        return 'indexed_text_lemma'
    if 'search_content' in df.columns and df['search_content'].astype(str).str.strip().any():
        return 'search_content'
    # fallback heuristics
    for candidate in ['long_description', 'description', 'title']:
        if candidate in df.columns:
            return candidate
    # pick first column
    return df.columns[0]

def build_tfidf_index(df: pd.DataFrame, field: str):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(df[field].astype(str).fillna('').values)
    return vec, mat

def query_search_from_index(query: str,
                            df: pd.DataFrame,
                            vectorizer: TfidfVectorizer,
                            tfidf_matrix,
                            text_field: str,
                            top_k: int):
    """
    Returns a DataFrame with:
      clean_text, product_title, url, score, sku,
      price_current, price_original, discount_percent, stock_status,
      summary, long_description
    """
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    idxs = sims.argsort()[::-1][:top_k]
    rows = []
    for i in idxs:
        row = df.iloc[i]
        rows.append({
            "clean_text": row.get("search_content", "") or row.get(text_field, ""),
            "product_title": row.get("title", ""),
            "url": row.get("product_url", "") or row.get("url", ""),
            "score": float(sims[i]),
            "sku": row.get("sku", ""),
            "price_current": row.get("price_current", ""),
            "price_original": row.get("price_original", ""),
            "discount_percent": row.get("discount_percent", ""),
            "stock_status": row.get("stock_status", ""),
            "summary": row.get("summary", ""),
            "long_description": row.get("long_description", "")
        })
    return pd.DataFrame(rows)

def extractive_summary_from_retrieved(retrieved_df, top_k=TOP_K_DEFAULT):
    if retrieved_df is None or retrieved_df.empty:
        return ""
    merged = " ".join(str(x) for x in retrieved_df["clean_text"].astype(str).head(top_k).tolist())
    merged = re.sub(r"[^a-zA-Z0-9\s\.\,\-]", " ", merged)
    sents = re.split(r"(?<=[.!?])\s+", merged)
    return " ".join(sents[:2]).strip()

# ---------- Intent detection (price/discount/stock/detail/general) ----------
def detect_question_intent(query: str) -> str:
    """
    Returns one of: 'discount', 'orig_price', 'price', 'stock', 'detail', 'general'
    """
    q = query.lower()
    # discount / offer
    if any(kw in q for kw in ["discount", "offer", "% off", "off "]):
        return "discount"
    # original price / mrp
    if any(kw in q for kw in ["original price", "mrp", "actual price", "before discount"]):
        return "orig_price"
    # current price / price / cost
    if any(kw in q for kw in ["price", "cost", "rate"]):
        return "price"
    # stock / availability / status
    if any(kw in q for kw in ["stock", "in stock", "out of stock", "available", "availability", "status"]):
        return "stock"
    # more details / explanation
    detail_phrases = [
        "more detail", "more details", "more in detail",
        "tell me more", "explain", "explanation",
        "full description", "describe", "more info", "more about this"
    ]
    if any(p in q for p in detail_phrases):
        return "detail"
    return "general"

def looks_like_followup(query: str, has_previous_product: bool) -> bool:
    """
    Decide if this question should use the last product context.
    We only treat it as follow-up if it is short / obviously referring to
    previous product (price, discount, stock, 'this', 'that', 'then', etc.).
    """
    if not has_previous_product:
        return False

    q = query.strip().lower()
    if not q:
        return False

    # obvious conversational starters
    if q.startswith("then ") or q.startswith("what about") or q.startswith("and "):
        return True

    # generic follow-up keywords (includes discount + offer)
    follow_keywords = [
        "price", "cost", "rate",
        "discount", "offer",
        "color", "colour", "size",
        "details", "more about", "explain",
        "stock", "availability", "status"
    ]
    if any(kw in q for kw in follow_keywords) and len(q.split()) <= 7:
        return True

    # references to previous / above product
    if ("above" in q or "previous" in q or "earlier" in q) and any(
        w in q for w in ["product", "item", "one"]
    ):
        return True

    # vague references, still treat as follow-up if short
    if any(w in q for w in ["this", "that", "it", "above", "previous", "earlier"]) and len(q.split()) <= 10:
        return True

    # super short like "Price?" / "Discount?" / "Stock?"
    if len(q.split()) <= 2 and detect_question_intent(query) in ["price", "discount", "stock", "orig_price"]:
        return True

    return False

# ---------- Main handler (stateless per-request) ----------
def handle_query_with_uploaded_csv(df: pd.DataFrame, query: str, top_k: int = TOP_K_DEFAULT):
    """
    df: uploaded dataframe (already read from CSV)
    query: user query string
    returns: dict with structure {query, top_results, final_answer, product_links}
    """
    fallback_msg = (
        "Sorry, I couldn't answer that. "
        "I can assist you with product, website, business or item-related queries."
    )

    if is_irrelevant(query):
        return {
            "query": query,
            "top_results": [],
            "final_answer": fallback_msg,
            "product_links": []
        }

    # choose field and index
    text_field = choose_text_field(df)
    vectorizer, tfidf_matrix = build_tfidf_index(df, text_field)

    # retrieval
    retrieved = query_search_from_index(
        query=query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        text_field=text_field,
        top_k=top_k
    )

    if retrieved.empty:
        return {
            "query": query,
            "top_results": [],
            "final_answer": fallback_msg,
            "product_links": []
        }

    best_score = float(retrieved["score"].max())

    # --- smart override: if query string is clearly matching a product title, don't fallback ---
    q_lower = query.lower()
    has_strong_title_match = False
    for _, r in retrieved.iterrows():
        title = (r.get("product_title") or "").lower()
        if title and q_lower in title:
            has_strong_title_match = True
            break

    if best_score < SIMILARITY_THRESHOLD and not has_strong_title_match:
        return {
            "query": query,
            "top_results": safe_jsonify(retrieved.to_dict(orient="records")),
            "final_answer": fallback_msg,
            "product_links": []
        }

    # ---------- Column-aware answering ----------
    intent = detect_question_intent(query)
    top = retrieved.iloc[0]

    title = (top.get("product_title") or "").strip()
    price_current = str(top.get("price_current") or "").strip()
    price_original = str(top.get("price_original") or "").strip()
    discount_percent = str(top.get("discount_percent") or "").strip()
    stock_status = str(top.get("stock_status") or "").strip()
    summary_txt = str(top.get("summary") or "").strip()
    long_desc = str(top.get("long_description") or "").strip()

    final_answer = None

    # 1) Price-type questions
    if intent == "price":
        if price_current and price_current.lower() != "nan":
            final_answer = f"The current price of {title} is {price_current}."
            if discount_percent and discount_percent.lower() not in ["nan", ""]:
                final_answer += f" It currently has a discount of {discount_percent}%."
        elif price_original and price_original.lower() != "nan":
            final_answer = (
                f"The price information of {title} is not fully available, "
                f"but the original price is {price_original}."
            )

    elif intent == "orig_price":
        if price_original and price_original.lower() != "nan":
            final_answer = f"The original price (before discount) of {title} is {price_original}."
        elif price_current and price_current.lower() != "nan":
            final_answer = (
                f"The original price is not available, "
                f"but the current price of {title} is {price_current}."
            )

    elif intent == "discount":
        if discount_percent and discount_percent.lower() != "nan" and discount_percent != "":
            final_answer = f"{title} currently has a discount of {discount_percent}%."
        elif price_original and price_current and \
             price_original.lower() != "nan" and price_current.lower() != "nan":
            try:
                po = float(price_original)
                pc = float(price_current)
                if po > 0:
                    disc = round((po - pc) / po * 100, 1)
                    final_answer = f"{title} has an approximate discount of {disc}%."
            except Exception:
                pass

    elif intent == "stock":
        if stock_status and stock_status.lower() != "nan":
            final_answer = f"{title} is currently {stock_status}."
        else:
            final_answer = f"The stock status of {title} is not clearly available."

    # 2) Detail questions → use long_description
    elif intent == "detail":
        if long_desc and long_desc.lower() != "nan":
            final_answer = f"Here are more details about {title}: {long_desc}"
        elif summary_txt:
            final_answer = f"Here is a summary of {title}: {summary_txt}"

    # 3) General questions → use summary
    if intent == "general" and not final_answer:
        if summary_txt and summary_txt.lower() != "nan":
            final_answer = f"{title}: {summary_txt}"
        else:
            final_answer = extractive_summary_from_retrieved(retrieved, top_k=top_k) or fallback_msg

    # Safety fallback
    if not final_answer:
        if summary_txt:
            final_answer = f"{title}: {summary_txt}"
        else:
            final_answer = extractive_summary_from_retrieved(retrieved, top_k=top_k) or fallback_msg

    product_links = [
        r.get("url") for _, r in retrieved.head(top_k).iterrows()
        if r.get("url")
    ]

    out = {
        "query": query,
        "top_results": safe_jsonify(retrieved.to_dict(orient="records")),
        "final_answer": final_answer,
        "product_links": product_links
    }
    return out

# ---------- Flask + SocketIO app ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)  # WSS-ready behind HTTPS / reverse proxy

# Simple in-memory session store (per WebSocket connection)
SESSION_MEMORY = {}  # { sid: {"history": [ ... ]} }

# ---------- HTTP routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": int(time.time())})

# ---------- WebSocket handlers ----------
@socketio.on("connect")
def ws_connect():
    sid = request.sid
    print(f"[WS] Client connected: {sid}")
    # initialize empty history for this connection
    SESSION_MEMORY[sid] = {"history": []}
    emit("system", {"message": "Connected to query API WebSocket."})

@socketio.on("disconnect")
def ws_disconnect():
    sid = request.sid
    print(f"[WS] Client disconnected: {sid}")
    # clear history for this connection
    SESSION_MEMORY.pop(sid, None)

@socketio.on("reset_session")
def ws_reset_session(data=None):
    sid = request.sid
    SESSION_MEMORY[sid] = {"history": []}
    emit("system", {"message": "Session context reset."})

@socketio.on("query_with_inputs")
def ws_query_with_inputs(data):
    """
    WebSocket version of /query_with_inputs.

    Expected message (JSON):
    {
      "csv_text": "<raw CSV content as text>",
      "query": "What is the price of ...?",
      "top_k": 3,              # optional
      "request_id": "abc123"   # optional, echoed back
    }

    Emits single event:
      - "query_result": { success, result or error, request_id? }
    """
    request_id = None
    sid = request.sid  # current websocket connection id

    try:
        # ---------- 0) Basic validation ----------
        if not isinstance(data, dict):
            emit("query_result", {
                "success": False,
                "error": "Payload must be a JSON object"
            })
            return

        request_id = data.get("request_id")
        csv_text = data.get("csv_text")
        query_text = data.get("quer") or data.get("query") or data.get("q")

        if not csv_text or str(csv_text).strip() == "":
            emit("query_result", {
                "success": False,
                "error": "Missing 'csv_text' in message",
                "request_id": request_id
            })
            return

        if not query_text or str(query_text).strip() == "":
            emit("query_result", {
                "success": False,
                "error": "Missing 'query' (or 'quer'/'q') in message",
                "request_id": request_id
            })
            return

        try:
            top_k = int(data.get("top_k", TOP_K_DEFAULT))
        except Exception:
            top_k = TOP_K_DEFAULT

        # ---------- 1) Load CSV ----------
        df = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")

        # ---------- 2) Fetch previous context for this WS session ----------
        session_state = SESSION_MEMORY.get(sid, {"history": []})

        # if somehow something else got stored, reset it
        if not isinstance(session_state, dict):
            session_state = {"history": []}

        history = session_state.get("history")
        if not isinstance(history, list):
            history = []

        previous_query = None
        previous_product_title = None
        previous_sku = None

        if history:
            last_turn = history[-1]

            # Only trust last_turn if it's a dict
            if isinstance(last_turn, dict):
                previous_query = last_turn.get("query")

                # Try to fetch last product title + SKU from last result
                last_result = last_turn.get("result")
                if isinstance(last_result, dict):
                    top_results = last_result.get("top_results") or []
                    if isinstance(top_results, list) and len(top_results) > 0:
                        top0 = top_results[0]
                        if isinstance(top0, dict):
                            previous_product_title = (top0.get("product_title") or "").strip()
                            previous_sku = str(top0.get("sku") or "").strip()
                        else:
                            # top0 is not a dict; ignore it
                            previous_product_title = None
                            previous_sku = None
                else:
                    # last_result not dict; ignore context
                    previous_product_title = None
                    previous_sku = None
            else:
                # history[-1] is not dict; ignore and reset history
                history = []

        # ---------- 3) Decide if this is follow-up or fresh query ----------
        current_q = str(query_text)
        has_prev_product = bool(previous_product_title or previous_sku)
        is_followup = looks_like_followup(current_q, has_prev_product)

        if is_followup and has_prev_product:
            # Kaggle-style expansion: focus on the last product
            if previous_product_title and previous_sku:
                effective_query = f"{current_q.strip()} for product: {previous_product_title} (SKU: {previous_sku})"
            elif previous_product_title:
                effective_query = f"{current_q.strip()} for product: {previous_product_title}"
            else:
                effective_query = f"{current_q.strip()} for product with SKU: {previous_sku}"
        else:
            # Treat as a new topic / new product
            effective_query = current_q

        # ---------- 4) Run handler on effective query ----------
        result = handle_query_with_uploaded_csv(
            df=df,
            query=effective_query,
            top_k=top_k
        )

        # ---------- 5) Update session history (keep last 5 turns) ----------
        history.append({
            "query": current_q,
            "effective_query": effective_query,
            "result": result,
            "timestamp": time.time()
        })
        if len(history) > 5:
            history = history[-5:]

        SESSION_MEMORY[sid] = {"history": history}

        # ---------- 6) Emit back to client ----------
        emit("query_result", {
            "success": True,
            "request_id": request_id,
            "result": safe_jsonify(result)
        })

    except Exception as e:
        emit("query_result", {
            "success": False,
            "request_id": request_id,
            "error": "Exception during query handling",
            "details": str(e)
        })

if __name__ == "__main__":
    print("Starting query API (HTTP + WebSocket) on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)