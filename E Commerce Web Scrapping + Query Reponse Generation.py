from flask import Flask, request
from flask_socketio import SocketIO, emit

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import os
import numpy as np
import json
from urllib.parse import urlparse, urljoin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from io import BytesIO, StringIO  # for CSV text in WS
from typing import Optional

# --- NEW: imports for scheduler ---
import schedule
import threading
from datetime import datetime

# ---------- Scheduler configuration ----------
# Weekly schedule configuration
# e.g. "thursday" and "11:05"  (24-hour format)
WEEKLY_DAY = "monday"   # monday, tuesday, wednesday, ...
WEEKLY_TIME = "00:00"   # "HH:MM" 24h

# This will be set by the user via /set-weekly-url or WS event
WEEKLY_SCRAPE_URL = None

# ---------- HTTP / scraping configuration ----------
REQUEST_DELAY = 1.0        # seconds between requests
TIMEOUT = 15               # seconds for HTTP timeout
USER_AGENT = 'Mozilla/5.0 (compatible; DataScraper/1.0)'
MAX_PAGES = 2000           # max pages per collection (safety limit)

# ---------- Ensure NLTK data (once at startup) ----------
nltk_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for r in nltk_data:
    try:
        nltk.data.find(r)
    except Exception:
        nltk.download(r)


# ---------- Helper utilities ----------
_currency_re = re.compile(r'[^\d.,\-]+')
MARKETING_WORDS = {
    'buy now','best','new','free shipping','hot','sale','discount','offer','trending'
}
COLOR_WORDS = {
    'black','white','red','blue','green','yellow','pink','orange','purple',
    'brown','grey','gray','silver','gold','navy'
}

# ---------- Config ----------
# UPLOAD_FOLDER = "./uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SIMILARITY_THRESHOLD = 0.20
TOP_K_DEFAULT = 3

# ---------- Helpers ----------
IRRELEVANT_KEYWORDS = [
    "gym", "dumbbell", "exercise", "workout", "recipe", "food", "cooking",
    "weather", "news", "politics", "relationship", "doctor", "medicine",
    "math", "code", "python", "java", "cpp"
]
_irrelevant_regex = re.compile(r'\b(' + r'|'.join(re.escape(w) for w in IRRELEVANT_KEYWORDS) + r')\b', flags=re.I)

def url_to_name(url: str) -> str:
    """
    Turn a URL into a safe base name:
      - strip common subdomains (www, m, app, shop)
      - combine domain + path parts
      - replace '-' and '.' with '_'
    """
    parsed = urlparse(url)
    netloc = parsed.netloc
    parts = netloc.split(".")
    if parts and parts[0] in ["www", "m", "app", "shop"]:
        parts = parts[1:]
    domain = parts[0] if parts else "site"

    # Get path parts and filter out empty strings
    path_parts = [p for p in parsed.path.split('/') if p]

    if path_parts:
        safe_path = "_".join(path_parts).replace('-', '_').replace('.', '_')
        return f"{domain}_{safe_path}"
    else:
        return domain

def resolve_csv_for_url(page_url: str, output_dir: str) -> Optional[str]:
    """
    Resolve which CSV to use for a given Shopify page URL.

    Priority:
      1. Per-collection CSV under: outputs_API_1/<site>/<collection_name>.csv
      2. Fallback: outputs_API_1/<site_name>_combined.csv
    """

    try:
        parsed = urlparse(page_url)
    except Exception:
        return None

    if not parsed.scheme or not parsed.netloc:
        return None

    # ---------------------------------------
    # Extract site name (domain only)
    # ---------------------------------------
    site_root = f"{parsed.scheme}://{parsed.netloc}/"
    site_name = url_to_name(site_root)  # e.g. "99wholesale"

    # ---------------------------------------
    # Per-site folder
    # outputs_API_1/<site_name>/
    # ---------------------------------------
    site_folder = os.path.join(output_dir, site_name)

    # ---------------------------------------
    # If URL contains /collections/<name>
    # Extract collection name
    # ---------------------------------------
    path_parts = parsed.path.strip("/").split("/")
    collection_csv_candidates = []

    if "collections" in path_parts:
        idx = path_parts.index("collections")
        if idx + 1 < len(path_parts):
            raw_collection_name = path_parts[idx + 1]  # "viral-gadgets"

            # Normalize collection name like your scraper does
            coll_name = url_to_name(f"{site_root}collections/{raw_collection_name}")

            # Expected per-collection CSV filenames
            collection_csv_candidates = [
                os.path.join(site_folder, f"{coll_name}.csv"),
                os.path.join(site_folder, f"{coll_name.replace('-', '_')}.csv"),
                os.path.join(site_folder, f"{coll_name.replace('_', '-')}.csv"),
            ]

            # Check if any per-collection CSV exists
            for f in collection_csv_candidates:
                if os.path.exists(f):
                    print(f"[resolve_csv_for_url] Using collection CSV: {f}")
                    return f

    # ---------------------------------------
    # Fallback to combined site CSV
    # ---------------------------------------
    combined_csv = os.path.join(output_dir, f"{site_name}_combined.csv")
    if os.path.exists(combined_csv):
        print(f"[resolve_csv_for_url] Using site combined CSV: {combined_csv}")
        return combined_csv

    print("[resolve_csv_for_url] No CSV found for:", page_url)
    return None

def parse_price(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s == '' or s.lower() in ['nan','none','null']:
        return np.nan
    s = _currency_re.sub('', s).replace(',', '')
    try:
        return float(s)
    except:
        nums = re.findall(r'[-+]?\d*\.\d+|\d+', s)
        return float(nums[0]) if nums else np.nan

def normalize_stock(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    if s in ['true','yes','1','in stock','available','instock']: return 'In Stock'
    if s in ['false','no','0','out of stock','sold out','not available']: return 'Out of Stock'
    return 'Unknown'

def clean_title(t):
    if pd.isna(t): return ''
    s = re.sub(r'\s+', ' ', str(t).strip())
    for w in MARKETING_WORDS:
        s = re.sub(r'(?i)\b' + re.escape(w) + r'\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def variant_looks_like_color(v):
    if pd.isna(v) or str(v).strip() == '': return False
    parts = re.split(r'[,/;|-]+', str(v).lower())
    return any(p.strip() in COLOR_WORDS for p in parts)

def clean_functional_text(txt):
    if pd.isna(txt): return ''
    s = re.sub(r'<[^>]+>', ' ', str(txt))
    s = re.sub(r'[\r\n\t]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    words = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum())

def extractive_summary(text, n_sentences=1):
    if not isinstance(text, str) or not text.strip():
        return ''
    sents = sent_tokenize(text)
    if len(sents) <= n_sentences:
        return ' '.join(sents)
    try:
        vec = TfidfVectorizer(stop_words='english')
        X = vec.fit_transform(sents)
        centroid = X.sum(axis=0)
        scores = X.dot(centroid.T).A.ravel()
        idx = scores.argsort()[::-1][:n_sentences]
        idx = sorted(idx)
        return ' '.join(sents[i].strip() for i in idx)
    except Exception:
        return sents[0]

def clean_url(u):
    if pd.isna(u): return ''
    s = str(u).strip()
    return re.sub(r'[\?&]$', '', s)

def looks_like_image_url(u):
    if pd.isna(u): return False
    return bool(re.search(r'\.(jpg|jpeg|png|webp|gif)$', str(u), flags=re.I))

# ---------- Scraper: single collection ----------

def scrape_shopify_products(base_collection_url, headers=None, sleep_sec=1, limit_per_page=250, max_pages=50):
    """
    Scrapes products.json using a collection URL as base.
    Returns pandas DataFrame with variant-level rows.
    """
    if headers is None:
        headers = {'User-Agent': USER_AGENT}

    # derive products.json endpoint (assume collection url + /products.json)
    if base_collection_url.endswith('/'):
        json_endpoint = base_collection_url + "products.json"
    else:
        json_endpoint = base_collection_url + "/products.json"

    all_variants = []
    page = 1
    while True:
        if page > max_pages:
            break
        url = f"{json_endpoint}?page={page}&limit={limit_per_page}"
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code != 200:
                # stop on non-200
                break
            data = resp.json()
            products = data.get('products', [])
            if not products:
                break

            for product in products:
                product_title = product.get('title','N/A')
                handle = product.get('handle','')
                vendor = product.get('vendor','N/A')
                category = product.get('product_type','N/A')
                raw_html = product.get('body_html','')
                functional_details = BeautifulSoup(raw_html,'html.parser').get_text(
                    separator=' ', strip=True
                ) if raw_html else ''
                tags_val = product.get('tags','')
                if isinstance(tags_val, list):
                    tags = ', '.join(tags_val)
                else:
                    tags = tags_val if tags_val else ''
                main_image_url = "N/A"
                if product.get('images'):
                    try:
                        if isinstance(product['images'], list) and product['images']:
                            main_image_url = product['images'][0].get('src','N/A')
                        else:
                            main_image_url = product['images']
                    except:
                        main_image_url = "N/A"

                parsed_collection_url = urlparse(base_collection_url)
                base_shop_url = f"{parsed_collection_url.scheme}://{parsed_collection_url.netloc}"

                for variant in product.get('variants', []):
                    variant_title = variant.get('title','N/A')
                    variant_id = variant.get('id')
                    price = variant.get('price','N/A')
                    original_price = variant.get('compare_at_price','')
                    sku = variant.get('sku','N/A')
                    available = variant.get('available', False)
                    link = (
                        f"{base_shop_url}/products/{handle}?variant={variant_id}"
                        if handle and variant_id else base_collection_url
                    )
                    if original_price and original_price != price:
                        discount_info = f"Was {original_price}"
                    else:
                        discount_info = "No Discount"
                    all_variants.append({
                        'Collection URL': base_collection_url,
                        'Product Name': product_title,
                        'Variant Name': variant_title,
                        'SKU': sku,
                        'In Stock?': available,
                        'Price': price,
                        'Original Price': original_price if original_price else "",
                        'Discount Info': discount_info,
                        'Vendor (Brand)': vendor,
                        'Category': category,
                        'Tags': tags,
                        'Functional Details': functional_details,
                        'Link': link,
                        'Main Image URL': main_image_url
                    })
            page += 1
            time.sleep(sleep_sec)
        except Exception:
            break
    return pd.DataFrame(all_variants)

# ---------- Scraper: discover all collections & scrape them ----------

def discover_collection_urls(start_url: str) -> set:
    """
    Discover unique /collections/ URLs from the given start_url.
    """
    collection_urls = set()
    print(f"\nDiscovering collection links from: {start_url}")
    try:
        response = requests.get(
            start_url,
            headers={'User-Agent': USER_AGENT},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        start_url_parsed = urlparse(start_url)
        start_domain = start_url_parsed.netloc

        for a_tag in soup.find_all('a', href=True):
            link = urljoin(start_url, a_tag['href'])
            parsed_link = urlparse(link)

            # Same domain + contains '/collections/'
            if parsed_link.netloc == start_domain and '/collections/' in parsed_link.path:
                clean_link = (
                    parsed_link.scheme + "://" + parsed_link.netloc +
                    parsed_link.path.split('?')[0].split('#')[0]
                )
                if clean_link.endswith('/'):
                    clean_link = clean_link[:-1]
                collection_urls.add(clean_link)

        print(f"Found {len(collection_urls)} potential unique collection URLs.")
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {start_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while discovering collections: {e}")
    finally:
        time.sleep(REQUEST_DELAY)

    return collection_urls

def scrape_all_collections(start_url: str, base_save_dir: str):
    """
    High-level site scraper:
      - Finds all /collections/ links from start_url
      - Scrapes each collection via products.json
      - Saves per-collection CSVs under base_save_dir/<site_name>/
      - Returns combined DataFrame and the site-specific directory.
    """
    collection_urls = discover_collection_urls(start_url)
    all_collections_data = []

    # Ensure the base save directory exists
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"\nEnsured base save directory exists: {base_save_dir}")
    print(f"Starting scrape for {len(collection_urls)} collections...")

    if not collection_urls:
        print("No collection URLs found; nothing to scrape.")
        return pd.DataFrame(), None

    # Site-specific directory (e.g., 'gangslifestyle')
    base_url_name = url_to_name(start_url)
    site_specific_save_dir = os.path.join(base_save_dir, base_url_name)
    os.makedirs(site_specific_save_dir, exist_ok=True)
    print(f"Ensured site-specific save directory exists: {site_specific_save_dir}")

    for i, collection_url in enumerate(sorted(list(collection_urls))):
        collection_json_endpoint = f"{collection_url}/products.json"
        collection_name_for_logging = url_to_name(collection_url)
        print(f"\nProcessing collection {i + 1}/{len(collection_urls)}: {collection_name_for_logging}")
        print(f"  JSON Endpoint: {collection_json_endpoint}")

        current_collection_variants_data = []
        page = 1

        while True:
            if page > MAX_PAGES:
                print(f"  Reached MAX_PAGES={MAX_PAGES} for {collection_name_for_logging}. Stopping pagination.")
                break

            json_url = f'{collection_json_endpoint}?page={page}&limit=250'
            print(f"  Fetching page {page} for {collection_name_for_logging}: {json_url}")

            try:
                response = requests.get(
                    json_url,
                    headers={'User-Agent': USER_AGENT},
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                if 'products' not in data or not data['products']:
                    print(f"  No more products found for {collection_name_for_logging} on page {page}.")
                    break

                products = data['products']
                print(f"  Found {len(products)} products on page {page}.")

                for product in products:
                    try:
                        product_title = product.get('title', 'N/A')
                        handle = product.get('handle')
                        vendor = product.get('vendor', 'N/A')
                        category = product.get('product_type', 'N/A')

                        raw_html = product.get('body_html', '')
                        if raw_html:
                            soup = BeautifulSoup(raw_html, 'html.parser')
                            functional_details = soup.get_text(separator=' ', strip=True)
                        else:
                            functional_details = "N/A"

                        tags = ', '.join(product.get('tags', []))

                        main_image_url = "N/A"
                        if product.get('images'):
                            main_image_url = product['images'][0]['src']

                        parsed_collection_url = urlparse(collection_url)
                        base_shop_url = f"{parsed_collection_url.scheme}://{parsed_collection_url.netloc}"

                        for variant in product.get('variants', []):
                            variant_title = variant.get('title', 'N/A')
                            variant_id = variant.get('id')
                            price = variant.get('price', 'N/A')
                            original_price = variant.get('compare_at_price')
                            sku = variant.get('sku', 'N/A')
                            available = variant.get('available', False)

                            link = f"{base_shop_url}/products/{handle}?variant={variant_id}"

                            discount_info = "No Discount"
                            try:
                                p = float(price)
                                op = float(original_price) if original_price else None
                                if op is not None and op > p:
                                    discount_info = f"Was {original_price}"
                            except ValueError:
                                pass

                            current_collection_variants_data.append({
                                'Collection URL': collection_url,
                                'Product Name': product_title,
                                'Variant Name': variant_title,
                                'SKU': sku,
                                'In Stock?': available,
                                'Price': price,
                                'Original Price': original_price if original_price else "N/A",
                                'Discount Info': discount_info,
                                'Vendor (Brand)': vendor,
                                'Category': category,
                                'Tags': tags,
                                'Functional Details': functional_details,
                                'Link': link,
                                'Main Image URL': main_image_url
                            })

                    except Exception as e:
                        print(f"  Error parsing product '{product.get('title', 'Unknown')}' in {collection_name_for_logging}: {e}")

                page += 1
                time.sleep(REQUEST_DELAY)

            except requests.exceptions.HTTPError as e:
                print(f"  HTTP Error for {collection_name_for_logging} on page {page}: {e}. Stopping pagination.")
                break
            except requests.exceptions.RequestException as e:
                print(f"  Request Error for {collection_name_for_logging} on page {page}: {e}. Stopping pagination.")
                break
            except Exception as e:
                print(f"  Unexpected error for {collection_name_for_logging} on page {page}: {e}. Stopping pagination.")
                break

        if current_collection_variants_data:
            print(f"  Collected {len(current_collection_variants_data)} variants for {collection_name_for_logging}.")
            all_collections_data.extend(current_collection_variants_data)

            # Save per-collection CSV
            group_df = pd.DataFrame(current_collection_variants_data)
            collection_filename_base = url_to_name(collection_url)
            csv_filename = os.path.join(site_specific_save_dir, f'{collection_filename_base}.csv')
            group_df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"  Saved {len(group_df)} variants to '{csv_filename}'.")
        else:
            print(f"  No variants collected for {collection_name_for_logging}.")

    print(f"\nFinished scraping all collections. Total variants collected: {len(all_collections_data)}")

    if not all_collections_data:
        return pd.DataFrame(), site_specific_save_dir

    df_all_variants = pd.DataFrame(all_collections_data)
    return df_all_variants, site_specific_save_dir

# ---------- Preprocessing ----------

def preprocess_variants_df(df_raw, category_threshold=0.5, variant_color_combine=True, summary_sentences=1):
    if df_raw is None or df_raw.shape[0] == 0:
        return pd.DataFrame()
    df = df_raw.copy()
    col_map = {
        'Product Name': 'title',
        'Variant Name': 'variant',
        'SKU': 'sku',
        'In Stock?': 'instock',
        'Price': 'price',
        'Original Price': 'original_price',
        'Discount Info': 'discount_info',
        'Category': 'category',
        'Tags': 'tags',
        'Functional Details': 'functional',
        'Link': 'product_url',
        'Main Image URL': 'image_url'
    }
    working = pd.DataFrame()
    for short, long in col_map.items():
        if short in df.columns:
            working[long] = df[short].fillna('')
        else:
            working[long] = ''

    working['title'] = working['title'].apply(clean_title)

    if variant_color_combine:
        working['variant'] = working.get('variant','')
        working['title'] = working.apply(
            lambda r: f"{r['title']} (Color: {r['variant']})"
            if variant_looks_like_color(r['variant']) else r['title'],
            axis=1
        )

    working['sku'] = working['sku'].astype(str).str.strip()
    missing = working['sku'] == ''
    if missing.any():
        working.loc[missing, 'sku'] = [f"MISSINGSKU_{i}" for i in range(1, missing.sum()+1)]

    working['stock_status'] = working['instock'].apply(normalize_stock)
    working['price_parsed'] = working['price'].apply(parse_price)
    working['original_price_parsed'] = working['original_price'].apply(parse_price)

    def compute_prices(r):
        p = r['price_parsed']; o = r['original_price_parsed']
        if pd.isna(o) or o == 0: o = p
        if pd.isna(p) and not pd.isna(o): p = o
        if pd.isna(p) or pd.isna(o): disc = np.nan
        else:
            disc = 0 if o == p else round((o - p) / o * 100, 1) if o > p else 0
        return pd.Series([p, o, disc])

    working[['price_current','price_original','discount_percent']] = working.apply(compute_prices, axis=1)

    def keep(col):
        filled = (working[col].astype(str).str.strip() != '').sum()
        return (filled / len(working)) >= category_threshold

    if 'category' in working.columns and not keep('category'):
        working.drop(columns=['category'], inplace=True)
    if 'tags' in working.columns and not keep('tags'):
        working.drop(columns=['tags'], inplace=True)

    working['long_description'] = working['functional'].apply(clean_functional_text)
    working['summary'] = working['long_description'].apply(
        lambda t: extractive_summary(t, summary_sentences)
    )
    working['indexed_text_lemma'] = working.apply(
        lambda r: lemmatize_text(
            str(r.get('title','')) + ' ' +
            str(r.get('summary','')) + ' ' +
            str(r.get('long_description',''))
        ),
        axis=1
    )
    working['product_url'] = working['product_url'].apply(clean_url)

    img_frac = working['image_url'].apply(looks_like_image_url).mean() if 'image_url' in working.columns else 0
    if 'image_url' in working.columns and img_frac > 0.99:
        working.drop(columns=['image_url'], inplace=True)

    working['search_content'] = (
        working.get('title','') + " " +
        working.get('summary','') + " " +
        working.get('long_description','')
    )
    export_cols = [
        'sku','title','price_current','price_original','discount_percent',
        'stock_status','summary','long_description','search_content',
        'indexed_text_lemma','product_url'
    ]
    export_cols = [c for c in export_cols if c in working.columns]
    cleaned = working[export_cols].copy()
    return cleaned

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

    We treat it as follow-up if it is short / obviously referring to
    the previous product (price, discount, stock, 'this', 'that', 'product', etc.).
    """
    if not has_previous_product:
        return False

    q = query.strip().lower()
    if not q:
        return False

    # --- Explicit generic patterns like "price of the product" / "this product" ---
    generic_product_refs = [
        "price of the product",
        "price of this product",
        "price of that product",
        "what is the price of the product",
        "what is the price of this product",
        "what is the price of that product",
        "discount on this product",
        "discount on the product",
        "discount on that product",
        "is this product in stock",
        "stock of this product",
        "stock of the product",
    ]
    for pat in generic_product_refs:
        if pat in q:
            return True

    # If it mentions "product" together with price/discount/stock/detail words → likely follow-up
    if "product" in q and any(
        kw in q for kw in [
            "price", "cost", "rate",
            "discount", "offer", "more detail", "details", "explain",
            "stock", "availability", "available", "in stock", "out of stock",
            "details", "more about", "info", "specification", "specifications",
        ]
    ):
        return True

    # obvious conversational starters
    if q.startswith("then ") or q.startswith("what about") or q.startswith("and "):
        return True

    # generic follow-up keywords (includes discount + offer)
    follow_keywords = [
        "price", "cost", "rate",
        "discount", "offer",
        "color", "colour", "size",
        "details", "more about", "explain",
        "stock", "availability", "status", "available", "in stock", "out of stock",
    ]
    # slightly more lenient length limit (was 7)
    if any(kw in q for kw in follow_keywords) and len(q.split()) <= 9:
        return True

    # references to previous / above product
    if ("above" in q or "previous" in q or "earlier" in q) and any(
        w in q for w in ["product", "item", "one"]
    ):
        return True

    # vague references, still treat as follow-up if short
    if any(w in q for w in ["this", "that", "it", "above", "previous", "earlier"]) and len(q.split()) <= 12:
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

    # Only hard-fallback if absolutely nothing matches (score == 0) and no strong title hint
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

app = Flask(__name__)
app.config['OUTPUT_DIR'] = os.path.join(os.getcwd(), "csv files")
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)  # WSS-ready behind HTTPS / reverse proxy

# Simple in-memory session store (per WebSocket connection)
SESSION_MEMORY = {}  # { sid: {"history": [ ... ]} }

# ---------- Shared pipeline for HTTP + WS ----------

def run_scrape_pipeline(base_url: str):
    """
    Run full pipeline:
      - If base_url contains '/collections/': scrape that collection only
      - Else: treat as site root, discover all collections & scrape them all
      - Preprocess and write a combined CSV
      - (Additionally, site-root mode writes per-collection CSVs under outputs/<site_name>/)

    Returns (out_path, out_fname, row_count)
    Raises ValueError for user-facing errors, Exception for internal errors.
    """
    if not base_url:
        raise ValueError("No 'url' provided.")

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL provided: {base_url}")

    # Decide mode: single collection vs entire site
    is_collection = '/collections/' in parsed.path

    if is_collection:
        print(f"[Pipeline] Treating URL as single collection: {base_url}")
        raw_df = scrape_shopify_products(
            base_collection_url=base_url,
            headers={'User-Agent': USER_AGENT},
            sleep_sec=REQUEST_DELAY,
            max_pages=MAX_PAGES
        )
        site_specific_dir = None
    else:
        print(f"[Pipeline] Treating URL as site root (multi-collection): {base_url}")
        raw_df, site_specific_dir = scrape_all_collections(
            start_url=base_url,
            base_save_dir=app.config['OUTPUT_DIR']
        )

    if raw_df is None or raw_df.shape[0] == 0:
        raise ValueError("No data scraped from the endpoint. Check the URL or site settings.")

    cleaned = preprocess_variants_df(raw_df)

    name = url_to_name(base_url)
    out_fname = f"{name}.csv" if is_collection else f"{name}_combined.csv"
    out_path = os.path.join(app.config['OUTPUT_DIR'], out_fname)

    cleaned.to_csv(out_path, index=False, encoding='utf-8')
    print(f"[Pipeline] Saved preprocessed CSV: {out_path} (rows={len(cleaned)})")

    return out_path, out_fname, len(cleaned)

@socketio.on("health")
def ws_health():
    emit("health-result", {
        "status": "ok",
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    })

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

@socketio.on("run-scrape")
def ws_run_scrape(data):
    """
    WebSocket version of /run-scrape

    Expected message:
    {
      "url": "<site_or_collection_url>"
    }

    Emits:
      - "run-scrape-status"
      - "run-scrape-result"
    """
    base_url = None
    if isinstance(data, dict):
        base_url = data.get("url")

    if not base_url:
        emit("run-scrape-result", {
            "success": False,
            "error": "No 'url' provided in WebSocket message."
        })
        return

    emit("run-scrape-status", {
        "status": "started",
        "url": base_url
    })

    try:
        out_path, out_fname, row_count = run_scrape_pipeline(base_url)
        emit("run-scrape-result", {
            "success": True,
            "url": base_url,
            "file_name": out_fname,
            "file_path": out_path,  # internal path
            "rows": row_count
        })
    except ValueError as e:
        emit("run-scrape-result", {
            "success": False,
            "url": base_url,
            "error": str(e)
        })
    except Exception as e:
        emit("run-scrape-result", {
            "success": False,
            "url": base_url,
            "error": "Internal error in scraper pipeline",
            "details": str(e)
        })

@socketio.on("set-weekly-url")
def ws_set_weekly_url(data):
    """
    WebSocket version of /set-weekly-url

    Expected message:
    {
      "url": "https://..."
    }

    Emits:
      - "set-weekly-url-result"
    """
    global WEEKLY_SCRAPE_URL

    url = None
    if isinstance(data, dict):
        url = data.get("url")

    if not url:
        emit("set-weekly-url-result", {
            "success": False,
            "error": "Missing 'url' in message"
        })
        return

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        emit("set-weekly-url-result", {
            "success": False,
            "error": "Invalid URL provided",
            "url": url
        })
        return

    WEEKLY_SCRAPE_URL = url
    emit("set-weekly-url-result", {
        "success": True,
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    })

@socketio.on("query_with_inputs")
def ws_query_with_inputs(data):
    """
    WebSocket query handler.

    You can call it in TWO ways:

    1) Explicit CSV text (old behaviour)
       {
         "csv_text": "<raw CSV content as text>",
         "query": "What is the price of ...?",
         "top_k": 3,
         "request_id": "abc123"
       }

    2) URL-based (NEW behaviour, preferred in your use case)
       {
         "page_url": "https://shop-domain.com/collections/hoodies",
         // or "url": "https://shop-domain.com/collections/hoodies",
         "query": "What is the cheapest hoodie?",
         "top_k": 3,
         "request_id": "abc123"
       }

    Emits:
      - "query_result": { success, result or error, request_id?, meta? }
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

        # NEW: URL-based lookup
        page_url = data.get("page_url") or data.get("url")

        # Old way: raw CSV text from client
        csv_text = data.get("csv_text")

        query_text = data.get("quer") or data.get("query") or data.get("q")

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

        # ---------- 1) Decide how to get the CSV ----------
        if csv_text and str(csv_text).strip() != "":
            # Old behaviour: client sends CSV content directly
            df = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")
            csv_source = "inline"
            csv_path_used = None

        elif page_url:
            # NEW: resolve CSV from URL (might be combined OR per-collection)
            csv_path = resolve_csv_for_url(page_url, app.config["OUTPUT_DIR"])
            if not csv_path:
                emit("query_result", {
                    "success": False,
                    "error": "No CSV found for this URL. Run /run-scrape first for this site.",
                    "page_url": page_url,
                    "request_id": request_id
                })
                return

            # Load raw CSV
            raw_df = pd.read_csv(csv_path, dtype=str)

            # ---------- IMPORTANT PART ----------
            # If this looks like RAW scraper output (has 'Product Name' etc. but NO 'search_content'),
            # then preprocess it into the format expected by the query engine.
            if ("search_content" not in raw_df.columns) and ("indexed_text_lemma" not in raw_df.columns):
                # assume this is the per-collection raw CSV from scrape_all_collections
                df = preprocess_variants_df(raw_df)
            else:
                # already preprocessed (e.g., <site_name>_combined.csv from run_scrape_pipeline)
                df = raw_df

            # Make sure there are no NaNs for downstream
            df = df.fillna("")

            csv_source = "file"
            csv_path_used = csv_path

        else:
            emit("query_result", {
                "success": False,
                "error": "Provide either 'csv_text' or 'page_url'/'url' in the message",
                "request_id": request_id
            })
            return

        # ---------- 2) Fetch previous context for this WS session ----------
        session_state = SESSION_MEMORY.get(sid, {"history": []})

        # if somehow something else got stored, reset it
        if not isinstance(session_state, dict):
            session_state = {"history": []}

        history = session_state.get("history")
        if not isinstance(history, list):
            history = []

        previous_product_title = None
        previous_sku = None

        if history:
            last_turn = history[-1]

            if isinstance(last_turn, dict):
                last_result = last_turn.get("result")
                if isinstance(last_result, dict):
                    top_results = last_result.get("top_results") or []
                    if isinstance(top_results, list) and len(top_results) > 0:
                        top0 = top_results[0]
                        if isinstance(top0, dict):
                            previous_product_title = (top0.get("product_title") or "").strip()
                            previous_sku = str(top0.get("sku") or "").strip()

        # ---------- 3) Decide if this is follow-up or fresh query ----------
        current_q = str(query_text)
        has_prev_product = bool(previous_product_title or previous_sku)
        is_followup = looks_like_followup(current_q, has_prev_product)

        # --- NEW OVERRIDE: if it looks like a *new product name*, don't treat as follow-up ---
        words = current_q.strip().lower().split()
        explicit_ref_words = ["this", "that", "it", "product", "above", "previous", "earlier", "same"]
        has_explicit_ref = any(w in current_q.lower() for w in explicit_ref_words)

        # If:
        #  - we initially thought it's follow-up
        #  - BUT the user didn't say "this/that/product/above/previous"
        #  - AND the question is longer than 3 words (so not just "Price?" / "Discount?")
        # then: assume they're talking about a NEW product by name.
        if is_followup and not has_explicit_ref and len(words) > 3:
            is_followup = False

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

        # Meta info (nice for debugging on client)
        result_meta = {
            "csv_source": csv_source,      # "inline" or "file"
            "csv_path": csv_path_used,     # full path on server if file-based
            "page_url": page_url
        }

        # ---------- 5) Update session history (keep last 5 turns) ----------
        history.append({
            "query": current_q,
            "effective_query": effective_query,
            "result": result,
            "csv_meta": result_meta,
            "timestamp": time.time()
        })
        if len(history) > 5:
            history = history[-5:]

        SESSION_MEMORY[sid] = {"history": history}

        # ---------- 6) Emit back to client ----------
        emit("query_result", {
            "success": True,
            "request_id": request_id,
            "result": safe_jsonify(result),
            "meta": result_meta
        })

    except Exception as e:
        emit("query_result", {
            "success": False,
            "request_id": request_id,
            "error": "Exception during query handling",
            "details": str(e)
        })

def weekly_scrape_job():
    """
    Weekly background job:
     - Runs run_scrape_pipeline(WEEKLY_SCRAPE_URL)
     - Saves combined preprocessed CSV (and per-collection CSVs in site-root mode)
     - Broadcasts result via WebSocket event "weekly-scrape-result"
    """
    global WEEKLY_SCRAPE_URL

    now = datetime.now()
    print(f"\n--- Weekly scheduler tick at {now} ---")

    if not WEEKLY_SCRAPE_URL:
        msg = "WEEKLY_SCRAPE_URL is not set yet. Skipping."
        print("  [Weekly]", msg)
        socketio.emit("weekly-scrape-result", {
            "success": False,
            "message": msg,
            "timestamp": now.isoformat()
        }, broadcast=True)
        return

    print(f"  [Weekly] Starting pipeline for {WEEKLY_SCRAPE_URL}")
    try:
        out_path, out_fname, row_count = run_scrape_pipeline(WEEKLY_SCRAPE_URL)
        print(f"  [Weekly] Saved {row_count} rows to {out_path}")

        socketio.emit("weekly-scrape-result", {
            "success": True,
            "url": WEEKLY_SCRAPE_URL,
            "file_name": out_fname,
            "file_path": out_path,
            "rows": row_count,
            "timestamp": now.isoformat()
        }, broadcast=True)
    except Exception as e:
        msg = f"Error during weekly scrape: {e}"
        print("  [Weekly]", msg)
        socketio.emit("weekly-scrape-result", {
            "success": False,
            "message": msg,
            "url": WEEKLY_SCRAPE_URL,
            "timestamp": now.isoformat()
        }, broadcast=True)

def start_scheduler():
    """
    Start a background thread that runs the weekly schedule.
    """
    day = WEEKLY_DAY.lower()
    job = getattr(schedule.every(), day)
    job.at(WEEKLY_TIME).do(weekly_scrape_job)

    print(f"Scheduler: configured weekly job on {WEEKLY_DAY} at {WEEKLY_TIME}")

    def run_loop():
        while True:
            schedule.run_pending()
            time.sleep(60)

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

if __name__ == '__main__':
    start_scheduler()
    print("Starting combined Web Scrapper + query API server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
