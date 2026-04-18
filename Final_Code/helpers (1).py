# helpers.py
# Framework-agnostic helpers ONLY

import hashlib
import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

import base64

# =========================================================
# -------------------- CONFIG ------------------------------
# =========================================================

REQUEST_DELAY = 1.0
TIMEOUT = 15
# -------------------------------
# Global request headers
# -------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


MAX_PAGES = 2000

SIMILARITY_THRESHOLD = 0.20
TOP_K_DEFAULT = 3

MARKETING_WORDS = {
    "buy now", "best", "new", "free shipping", "hot",
    "sale", "discount", "offer", "trending"
}

COLOR_WORDS = {
    "black","white","red","blue","green","yellow","pink",
    "orange","purple","brown","grey","gray","silver","gold","navy"
}

IRRELEVANT_KEYWORDS = [
    "gym","dumbbell","exercise","workout","recipe","food",
    "weather","news","politics","doctor","medicine",
    "math","code","python","java","cpp"
]

_irrelevant_regex = re.compile(
    r"\b(" + r"|".join(re.escape(w) for w in IRRELEVANT_KEYWORDS) + r")\b",
    flags=re.I
)

_currency_re = re.compile(r"[^\d.,\-]+")


# =========================================================
# -------------------- NLTK -------------------------------
# =========================================================

def ensure_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg)


ensure_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# =========================================================
# -------------------- URL / CSV ---------------------------
# =========================================================

def path_to_base64(path: str) -> str:
    """
    Convert an absolute directory path into Base64 string.
    """
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    return base64.b64encode(abs_path.encode("utf-8")).decode("utf-8")

def is_shopify_site(url: str) -> bool:
    """
    Safely detect if a site is Shopify.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        html = r.text.lower()

        # quick signal
        if "cdn.shopify.com" not in html and not any(
            "shopify" in v.lower() for v in r.headers.values()
        ):
            return False

        # hard validation
        test = requests.get(
            url.rstrip("/") + "/products.json?limit=1",
            headers=HEADERS,
            timeout=TIMEOUT
        )

        if test.status_code != 200:
            return False

        data = test.json()
        return isinstance(data, dict) and "products" in data

    except Exception:
        return False

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
    combined_csv = os.path.join(
        output_dir,
        site_name,
        f"{site_name}_combined.csv"
    )

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

# ---------- Non - Shopify Scrapper ----------

def generate_sku(url):
    return "HTMLSKU_" + hashlib.md5(url.encode()).hexdigest()[:10]

PRICE_REGEX = re.compile(r"(₹|\$|€)\s?\d+(?:,\d+)*(?:\.\d+)?")

def extract_price(text):
    m = PRICE_REGEX.search(text)
    return m.group(0) if m else "N/A"

def detect_stock(text):
    t = text.lower()
    if "out of stock" in t or "sold out" in t:
        return "Out of Stock"
    if "in stock" in t or "available" in t:
        return "In Stock"
    return "Unknown"

def extract_image(soup, base_url):
    img = soup.find("img")
    return urljoin(base_url, img["src"]) if img and img.get("src") else "N/A"

def extract_tags(soup):
    meta = soup.find("meta", attrs={"name": "keywords"})
    return meta.get("content") if meta else "N/A"

def scrape_non_shopify_site(start_url: str, output_dir: str):
    print("\n🚀 NON-SHOPIFY SCRAPER STARTED")

    parsed = urlparse(start_url)
    base_domain = parsed.netloc
    site_name = base_domain.replace(".", "_")

    site_dir = os.path.join(output_dir, site_name)
    os.makedirs(site_dir, exist_ok=True)

    visited = set()
    queue = deque([start_url])
    all_rows = []

    while queue and len(visited) < MAX_PAGES:
        current_url = queue.popleft()
        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Scraping → {current_url}")

        try:
            r = requests.get(
                current_url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT
            )
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script", "style", "noscript"]):
                t.decompose()

            # ---------------- PAGE DATA ----------------
            h1 = soup.find("h1")
            product_title = (
                h1.get_text(strip=True)
                if h1 else soup.title.string.strip()
                if soup.title else "N/A"
            )

            full_text = soup.get_text(" ", strip=True)
            price = extract_price(full_text)
            stock = detect_stock(full_text)
            image_url = extract_image(soup, current_url)
            tags = extract_tags(soup)

            path_parts = [p for p in urlparse(current_url).path.split("/") if p]
            category = path_parts[0] if path_parts else "N/A"

            row = {
                "Collection URL": start_url,
                "Product Name": product_title,
                "Variant Name": "N/A",
                "SKU": generate_sku(current_url),
                "In Stock?": stock,
                "Price": price,
                "Original Price": "N/A",
                "Discount Info": "N/A",
                "Vendor (Brand)": base_domain,
                "Category": category,
                "Tags": tags,
                "Functional Details": full_text[:3000],
                "Link": current_url,
                "Main Image URL": image_url
            }

            # ---------- SAVE PAGE CSV ----------
            page_df = pd.DataFrame([row])
            page_csv = os.path.join(site_dir, f"{url_to_name(current_url)}.csv")
            page_df.to_csv(page_csv, index=False, encoding="utf-8")

            all_rows.append(row)

            # ---------- LINK DISCOVERY ----------
            for a in soup.find_all("a", href=True):
                link = urljoin(current_url, a["href"])
                parsed_link = urlparse(link)
                if parsed_link.netloc == base_domain:
                    clean = parsed_link.scheme + "://" + parsed_link.netloc + parsed_link.path
                    if clean not in visited:
                        queue.append(clean)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"Failed → {current_url} | {e}")

    # =====================================================
    # COMBINE ALL PAGES
    # =====================================================
    if all_rows:

        print("\n✅ SCRAPING COMPLETED")
        print(f"Pages scraped: {len(all_rows)}")
    else:
        print("❌ No pages scraped")
    
    return pd.DataFrame(all_rows)

# ---------- Shopify Scrapper ----------
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
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
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
    # if 'image_url' in working.columns and img_frac > 0.99:
    #     working.drop(columns=['image_url'], inplace=True)

    working['search_content'] = (
        working.get('title','') + " " +
        working.get('summary','') + " " +
        working.get('long_description','')
    )
    export_cols = [
        'sku','title','price_current','price_original','discount_percent',
        'stock_status','summary','long_description','search_content',
        'indexed_text_lemma','product_url','image_url'
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
      clean_text, product_title, url, image_url, score, sku,
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
            "image_url": row.get("image_url", ""),   # ✅ ADD THIS
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
    
    if any(k in q for k in ["url", "link", "product url", "product link"]):
        return "url"

    if any(k in q for k in ["image", "photo", "picture"]):
        return "image"
    
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
        "url", "link", "product url", "product link",
        "image", "photo", "picture"
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

def run_scrape_pipeline(base_url: str, output_dir: str):
    is_collection = False
    """
    Framework-agnostic scrape pipeline.

    Returns:
        (site_dir, out_fname, row_count)
    """
    if not base_url:
        raise ValueError("No 'url' provided.")

    if not output_dir:
        raise ValueError("No 'output_dir' provided.")

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL provided: {base_url}")

    os.makedirs(output_dir, exist_ok=True)

    # ======================================================
    # Decide scraper type
    # ======================================================
    shopify = is_shopify_site(base_url)

    if shopify:
        print(f"[Pipeline] Shopify site detected → {base_url}")

        is_collection = "/collections/" in parsed.path

        if is_collection:
            raw_df = scrape_shopify_products(
                base_collection_url=base_url,
                headers=HEADERS,
                sleep_sec=REQUEST_DELAY,
                max_pages=MAX_PAGES
            )
        else:
            raw_df, _ = scrape_all_collections(
                start_url=base_url,
                base_save_dir=output_dir
            )

    else:
        print(f"[Pipeline] Non-Shopify site detected → {base_url}")

        raw_df = scrape_non_shopify_site(
            start_url=base_url,
            output_dir=output_dir
        )

    if raw_df is None or raw_df.empty:
        raise ValueError("No data scraped from the endpoint.")

    cleaned = preprocess_variants_df(raw_df)
    if cleaned.empty:
        raise ValueError("No usable rows after preprocessing.")

    # 🔹 CREATE SITE-SPECIFIC FOLDER (AFTER SCRAPING)
    site_slug = url_to_name(base_url)
    site_dir = os.path.join(output_dir, site_slug)
    os.makedirs(site_dir, exist_ok=True)

    out_fname = f"{site_slug}.csv" if is_collection else f"{site_slug}_combined.csv"
    out_path = os.path.join(site_dir, out_fname)

    cleaned.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[Pipeline] Saved CSV → {out_path} ({len(cleaned)} rows)")

    # 🔁 RETURN SITE DIR (for Base64)
    return site_dir, out_fname, len(cleaned)
