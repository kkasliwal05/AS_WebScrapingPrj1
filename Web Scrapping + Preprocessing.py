"""
Flask API + WebSocket wrapper for:
 - Shopify /products.json variant-level scraper
 - Preprocessing pipeline (cleaning + export)
 - Weekly scheduled scraping job (URL configurable via JSON API or WebSocket)
 - NEW: Site-level scraper that:
      * Discovers all /collections/ links from a start URL
      * Scrapes each collection's /products.json
      * Saves per-collection CSVs + combined preprocessed CSV
"""

from flask import Flask, request, jsonify, send_file
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

# ---------- Flask + SocketIO setup ----------

app = Flask(__name__)
app.config['OUTPUT_DIR'] = os.path.join(os.getcwd(), "outputs_API_1")
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")  # WSS-ready behind HTTPS

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

# ---------- HTTP Endpoints ----------

@app.route('/run-scrape', methods=['POST'])
def run_scrape_http():
    """
    On-demand scrape via HTTP.
    Accepts:
      - JSON body: {"url": "<site_or_collection_url>"}
      - OR multipart form-data with file field 'config_file' (JSON)
      - OR form field 'url'
    """
    base_url = None

    if request.is_json:
        body = request.get_json()
        base_url = body.get('url') if isinstance(body, dict) else None

    if not base_url and 'config_file' in request.files:
        try:
            f = request.files['config_file']
            contents = f.read().decode('utf-8')
            cfg = json.loads(contents)
            base_url = cfg.get('url')
        except Exception as e:
            return jsonify({"error": "Failed to parse uploaded config file", "details": str(e)}), 400

    if not base_url and request.form.get('url'):
        base_url = request.form.get('url')

    if not base_url:
        return jsonify({"error": "No 'url' provided. Supply JSON body {\"url\":\"...\"} or upload config_file."}), 400

    try:
        out_path, out_fname, _ = run_scrape_pipeline(base_url)
        return send_file(out_path, as_attachment=True, download_name=out_fname)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal error", "details": str(e)}), 500

@app.route('/set-weekly-url', methods=['POST'])
def set_weekly_url_http():
    """
    Configure the URL used by the weekly scheduled scraper via HTTP.
    JSON body:
    {
      "url": "https://gangslifestyle.com/"
    }
    """
    global WEEKLY_SCRAPE_URL

    if not request.is_json:
        return jsonify({"error": "Expected JSON body with 'url'"}), 400

    body = request.get_json()
    url = body.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' in JSON body"}), 400

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return jsonify({"error": "Invalid URL provided", "url": url}), 400

    WEEKLY_SCRAPE_URL = url
    return jsonify({
        "message": "Weekly scrape URL updated successfully.",
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "weekly_url": WEEKLY_SCRAPE_URL,
        "day": WEEKLY_DAY,
        "time": WEEKLY_TIME
    })

# ---------- WebSocket Handlers ----------

@socketio.on("connect")
def ws_connect():
    print(f"WebSocket connected: {request.sid}")
    emit("system", {"message": "Connected to Shopify scraper WebSocket."})

@socketio.on("disconnect")
def ws_disconnect():
    print(f"WebSocket disconnected: {request.sid}")

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

# ---------- Weekly scheduled job ----------

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
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)