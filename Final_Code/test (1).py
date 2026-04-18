# # test.py
# import os
# import pandas as pd
# from uuid import uuid4

# from helpers import (
#     run_scrape_pipeline,
#     resolve_csv_for_url,
#     preprocess_variants_df,
#     choose_text_field,
#     build_tfidf_index,
#     query_search_from_index,
#     detect_question_intent,
#     looks_like_followup,
#     extractive_summary_from_retrieved
# )

# # --------------------------------------------------
# # Config
# # --------------------------------------------------
# OUTPUT_DIR = "./csv_files"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SESSION_ID = str(uuid4())
# TOKENS_LEFT = None
# ACTIVE_CHAT = False
# HISTORY = []


# # --------------------------------------------------
# # Helper functions
# # --------------------------------------------------
# def start_chat(token_init=None):
#     global TOKENS_LEFT, ACTIVE_CHAT, HISTORY
#     if token_init is not None and TOKENS_LEFT is None:
#         TOKENS_LEFT = max(0, int(token_init))
#     if TOKENS_LEFT is not None and TOKENS_LEFT <= 0:
#         print("❌ Tokens exhausted.")
#         return False
#     ACTIVE_CHAT = True
#     HISTORY.clear()
#     print(f"\n🟢 Chat started | Tokens left: {TOKENS_LEFT}")
#     return True


# def end_chat():
#     global TOKENS_LEFT, ACTIVE_CHAT, HISTORY
#     if TOKENS_LEFT is not None:
#         TOKENS_LEFT = max(0, TOKENS_LEFT - 1)
#     ACTIVE_CHAT = False
#     HISTORY.clear()
#     print(f"\n🔴 Chat ended | Tokens left: {TOKENS_LEFT}")


# def ask(page_url, query, top_k=3):
#     global HISTORY, ACTIVE_CHAT

#     if not ACTIVE_CHAT:
#         print("⚠️ Start a chat first.")
#         return

#     csv_path = resolve_csv_for_url(page_url, OUTPUT_DIR)
#     if not csv_path:
#         print("❌ CSV not found. Run scrape first.")
#         return

#     raw_df = pd.read_csv(csv_path, dtype=str).fillna("")
#     df = preprocess_variants_df(raw_df) \
#         if "search_content" not in raw_df.columns else raw_df

#     prev_product = HISTORY[-1]["top_results"][0] if HISTORY else None
#     has_prev = prev_product is not None

#     follow = looks_like_followup(query, has_prev)
#     intent = detect_question_intent(query)

#     if follow and prev_product:
#         effective_query = (
#             f"{query} for product "
#             f"{prev_product['product_title']} "
#             f"(SKU: {prev_product['sku']})"
#         )
#     else:
#         effective_query = query

#     field = choose_text_field(df)
#     vec, mat = build_tfidf_index(df, field)

#     res_df = query_search_from_index(
#         effective_query, df, vec, mat, field, top_k
#     )

#     results = res_df.to_dict(orient="records")
#     summary = extractive_summary_from_retrieved(res_df)

#     HISTORY.append({
#         "query": query,
#         "effective_query": effective_query,
#         "top_results": results
#     })
#     HISTORY[:] = HISTORY[-5:]

#     # ----------------- Output -----------------
#     print(f"\n🧠 Query: {query}")
#     print(f"🔍 Intent: {intent}")
#     print(f"↪ Follow-up: {follow}")
#     print(f"📄 Summary: {summary}")
#     print("📦 Results:")
#     for r in results:
#         print(
#             f" - {r['product_title']} | "
#             f"₹{r['price_current']} | "
#             f"{r['stock_status']}"
#         )


# # --------------------------------------------------
# # Main CLI flow
# # --------------------------------------------------
# if __name__ == "__main__":
#     print("\n====== Shopify Scraper CLI Test ======\n")

#     site_url = input("Enter Shopify site or collection URL: ").strip()

#     print("\n🔄 Running scraper...")
#     run_scrape_pipeline(site_url, OUTPUT_DIR)

#     token_init = int(input("\nEnter initial tokens (e.g. 3): ").strip())
#     if not start_chat(token_init):
#         exit(0)

#     while True:
#         q = input("\nAsk a question ('end' to end chat, 'exit' to quit): ").strip()
#         if q.lower() == "exit":
#             break
#         if q.lower() == "end":
#             end_chat()
#             if TOKENS_LEFT and TOKENS_LEFT > 0:
#                 start_chat()
#             continue
#         ask(site_url, q)


import base64

base64_value = "RDpcR2l0SHViXHNjcmFwLW5scC1tb2RlbFxGaW5hbGl6ZWRcY3N2X2ZpbGVzXGdhbmdzbGlmZXN0eWxl"

decoded_path = base64.b64decode(base64_value).decode("utf-8")

print(decoded_path)
