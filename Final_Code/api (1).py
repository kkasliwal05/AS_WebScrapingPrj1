# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import time
import threading
import schedule
from datetime import datetime
import pandas as pd
from urllib.parse import urlparse
import uuid
from helpers import (
    run_scrape_pipeline,
    resolve_csv_for_url,
    preprocess_variants_df,
    choose_text_field,
    build_tfidf_index,
    query_search_from_index,
    detect_question_intent,
    looks_like_followup,
    extractive_summary_from_retrieved,
    safe_jsonify,
    path_to_base64,
)
from fastapi.middleware.cors import CORSMiddleware

# ======================================================
# App setup
# ======================================================
app = FastAPI(title="Shopify Scraper + Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4200",
        "https://projectai.arinova.studio"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OUTPUT_DIR = os.path.join(os.getcwd(), "csv_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# In-memory session store
# ======================================================
# session_id -> { history, tokens_left, active_chat }
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ======================================================
# Weekly scrape configuration (GLOBAL)
# ======================================================
WEEKLY_SCRAPE_URL: Optional[str] = None
WEEKLY_DAY = "monday"     # monday, tuesday, ...
WEEKLY_TIME = "00:00"     # HH:MM (24h)


# ======================================================
# Request Models
# ======================================================
class ScrapeRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    session_id: str
    page_url: str
    query: str
    top_k: Optional[int] = 3
    token_init: Optional[int] = None
    end_chat: Optional[bool] = False


class WeeklyURLRequest(BaseModel):
    url: str

def detect_media_intent(query: str) -> Optional[str]:
    q = query.lower()

    image_words = [
        "image", "photo", "picture", "pic", "show image", "show photo"
    ]

    url_words = [
        "url", "link", "product url", "product link",
        "buy link", "product page", "open product",
        "where can i buy", "purchase link"
    ]

    if any(w in q for w in image_words):
        return "image"

    if any(w in q for w in url_words):
        return "url"

    return None

# ======================================================
# Answer formatter ⭐ (NEW)
# ======================================================
def format_answer(intent: str, product: Optional[dict], summary: str) -> str:
    if not product:
        return "Sorry, I couldn’t find a matching product."

    if intent == "price":
        return f"The price of this product is ₹{product.get('price_current')}."

    if intent == "discount":
        return f"This product has a {product.get('discount_percent')}% discount."

    if intent == "stock":
        return f"This product is currently {product.get('stock_status')}."

    if intent == "detail" or intent == "description" or intent == "summary":
        return product.get("summary") or summary

    if intent == "url":
        return product.get("url") or product.get("product_url") or "Product URL not available."

    if intent == "image":
        return product.get("image_url") or "Product image not available."

    return summary


# ======================================================
# Scrape Endpoint
# ======================================================
@app.post("/scrape")
def scrape(req: ScrapeRequest):
    try:
        site_dir, fname, rows = run_scrape_pipeline(
            base_url=req.url,
            output_dir=OUTPUT_DIR
        )

        base64_path = path_to_base64(site_dir)

        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        SESSIONS[session_id] = {
            "history": [],
            "tokens_left": None,
            "active_chat": False
        }

        # 🔑 base directory where all site CSVs are stored
        # site_dir = os.path.dirname(out_path)

        # # 🔐 Base64 encode the directory path
        # base_dir_base64 = path_to_base64(site_dir)

        return {
            "success": True,
            "session_id": session_id,
            "file_name": fname,
            "rows": rows,
            "base_dir_base64": base64_path  # ✅ ADDED
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================================================
# Query Endpoint (TOKENS + FOLLOW-UPS)
# ======================================================
@app.post("/query")
def query_products(req: QueryRequest):
    # -----------------------------
    # Session bootstrap
    # -----------------------------
    session = SESSIONS.setdefault(req.session_id, {
        "history": [],
        "tokens_left": None,
        "active_chat": False
    })

    history = session["history"]
    tokens_left = session["tokens_left"]
    active_chat = session["active_chat"]

    # -----------------------------
    # Token initialization (once)
    # -----------------------------
    if req.token_init is not None and tokens_left is None:
        tokens_left = max(0, int(req.token_init))

    # -----------------------------
    # New chat guard
    # -----------------------------
    if not active_chat:
        if tokens_left is not None and tokens_left <= 0:
            raise HTTPException(
                status_code=403,
                detail="Tokens exhausted. Please add tokens."
            )
        active_chat = True
        history.clear()

    # -----------------------------
    # Resolve CSV
    # -----------------------------
    csv_path = resolve_csv_for_url(req.page_url, OUTPUT_DIR)
    if not csv_path:
        raise HTTPException(
            status_code=404,
            detail="CSV not found. Run /scrape first."
        )

    raw_df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = (
        preprocess_variants_df(raw_df)
        if "search_content" not in raw_df.columns
        else raw_df
    )

    # -----------------------------
    # Follow-up handling
    # -----------------------------
    previous_product = None
    if history and history[-1]["top_results"]:
        previous_product = history[-1]["top_results"][0]

    has_prev = previous_product is not None
    follow = looks_like_followup(req.query, has_prev)
    intent = detect_question_intent(req.query)
    media_intent = detect_media_intent(req.query)


    # 🚫 If user asks URL or image but no previous product → stop here
    if media_intent and not previous_product:
        return {
            "success": True,
            "intent": media_intent,
            "follow_up": False,
            "answer": "Please tell me which product you are referring to.",
            "tokens_left": tokens_left,
            "chat_ended": False
        }

    if follow and previous_product:
        effective_query = (
            f"{req.query} for product "
            f"{previous_product['product_title']} "
            f"(SKU: {previous_product['sku']})"
        )
    else:
        effective_query = req.query

    # -----------------------------
    # Search
    # -----------------------------
    text_field = choose_text_field(df)
    vec, mat = build_tfidf_index(df, text_field)

    res_df = query_search_from_index(
        query=effective_query,
        df=df,
        vectorizer=vec,
        tfidf_matrix=mat,
        text_field=text_field,
        top_k=req.top_k
    )

    results = res_df.to_dict(orient="records")
    summary = extractive_summary_from_retrieved(res_df)

    # -----------------------------
    # Save history
    # -----------------------------
    history.append({
        "query": req.query,
        "effective_query": effective_query,
        "top_results": results
    })
    history[:] = history[-5:]

    # -----------------------------
    # End chat → consume token
    # -----------------------------
    chat_ended = False
    if req.end_chat:
        if tokens_left is not None:
            tokens_left = max(0, tokens_left - 1)
        history.clear()
        active_chat = False
        chat_ended = True

    # -----------------------------
    # Persist session
    # -----------------------------
    SESSIONS[req.session_id] = {
        "history": history,
        "tokens_left": tokens_left,
        "active_chat": active_chat
    }

    # ⭐ Build final answer
    answer = format_answer(
        intent,
        results[0] if results else None,
        summary
    )

    return {
        "success": True,
        "intent": intent,
        "follow_up": follow,
        "answer": answer,                 # ✅ ADD THIS
        "summary": summary,
        "top_results": safe_jsonify(results),
        "tokens_left": tokens_left,
        "chat_ended": chat_ended
    }



# ======================================================
# Weekly URL Route (ADMIN / SYSTEM)
# ======================================================
@app.post("/set-weekly-url")
def set_weekly_url(req: WeeklyURLRequest):
    global WEEKLY_SCRAPE_URL

    parsed = urlparse(req.url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL")

    WEEKLY_SCRAPE_URL = req.url

    return {
        "success": True,
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    }


@app.get("/weekly-status")
def weekly_status():
    return {
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    }


# ======================================================
# Weekly Scheduler
# ======================================================
def weekly_scrape_job():
    global WEEKLY_SCRAPE_URL

    now = datetime.now().isoformat()
    print(f"[Weekly] Tick @ {now}")

    if not WEEKLY_SCRAPE_URL:
        print("[Weekly] No weekly URL set. Skipping.")
        return

    try:
        out_path, fname, rows = run_scrape_pipeline(
            base_url=WEEKLY_SCRAPE_URL,
            output_dir=OUTPUT_DIR
        )
        print(f"[Weekly] Success: {rows} rows → {fname}")
    except Exception as e:
        print(f"[Weekly] Error: {e}")


def start_weekly_scheduler():
    day = WEEKLY_DAY.lower()
    job = getattr(schedule.every(), day)
    job.at(WEEKLY_TIME).do(weekly_scrape_job)

    print(f"[Scheduler] Weekly job set → {WEEKLY_DAY} @ {WEEKLY_TIME}")

    def run():
        while True:
            schedule.run_pending()
            time.sleep(60)

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ======================================================
# Start scheduler on app startup
# ======================================================
start_weekly_scheduler()
