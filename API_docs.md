
# E-Commerce Data Scraper + Query API

This service exposes:

- A **E-Commerce** (per-collection or whole site)
- A **query engine** over the generated CSVs
- A **weekly scheduler** that auto-runs the scraper
- A **WebSocket API** via Socket.IO

It is implemented with Flask + Flask-SocketIO and has **no REST endpoints** in this file — all interactions are via Socket.IO events.

---

## Overview

- **WebSocket base**: `ws://<host>:5000/socket.io`
- **Transport**: Socket.IO (WebSocket / polling)
- **Serialization**: JSON

Global settings (from code):

- `SIMILARITY_THRESHOLD = 0.20`
- `TOP_K_DEFAULT = 3`
- `OUTPUT_DIR = "<cwd>/csv files"`
- `WEEKLY_DAY = "monday"`
- `WEEKLY_TIME = "00:00"` (24h)
- `MAX_PAGES = 2000` (pagination safety limit)

The server maintains an in-memory, per-connection session:

```python
SESSION_MEMORY = {
  <sid>: {
    "history": [
      {
        "query": "...",
        "effective_query": "...",
        "result": { ... },
        "csv_meta": { ... },
        "timestamp": 1234567890.0
      },
      ...
    ]
  }
}
````

---

## Events Summary

### Client → Server

* `health`
* `reset_session`
* `run-scrape`
* `set-weekly-url`
* `query_with_inputs`

### Server → Client

* `system`
* `health-result`
* `run-scrape-status`
* `run-scrape-result`
* `set-weekly-url-result`
* `query_result`
* `weekly-scrape-result`

---

## 1. Connection & Session

### `connect` (server → client)

On successful WebSocket connection, the server:

* Creates a new session history: `SESSION_MEMORY[sid] = {"history": []}`
* Emits:

```json
event: "system"
data: {
  "message": "Connected to query API WebSocket."
}
```

### `disconnect` (server side)

On disconnect, the server:

* Deletes the session entry: `SESSION_MEMORY.pop(sid, None)`

### `reset_session` (client → server)

**Description:** Clear conversation context / follow-up history for the current WebSocket connection.

**Payload:** (any or none)

```json
{}
```

**Response (server → client):**

```json
event: "system"
data: {
  "message": "Session context reset."
}
```

---

## 2. Health Check

### `health` (client → server)

**Description:** Check if server is alive and see current weekly scheduler settings.

**Payload:**

```json
{}
```

**Response (server → client):**

```json
event: "health-result"
data: {
  "status": "ok",
  "weekly_url": "<string or null>",
  "day": "monday",
  "time": "00:00"
}
```

---

## 3. Scraper: Run On Demand

### `run-scrape` (client → server)

**Description:** Run the full scraper pipeline for a Shopify **site** or a **single collection**.

**Mode detection:**

* If `url` **contains** `/collections/` → treat as **single collection**
* Else → treat as **site root** and scrape all discovered collections

**Request payload:**

```json
{
  "url": "https://example-shop.com"
}
```

* `url` (string, required): Shopify root or collection URL.

**Intermediate status (server → client):**

```json
event: "run-scrape-status"
data: {
  "status": "started",
  "url": "https://example-shop.com"
}
```

**Success response (server → client):**

```json
event: "run-scrape-result"
data: {
  "success": true,
  "url": "https://example-shop.com",
  "file_name": "<generated CSV file name>",
  "file_path": "<absolute server path>",
  "rows": 1234
}
```

**Validation / user error:**

```json
event: "run-scrape-result"
data: {
  "success": false,
  "url": "https://example-shop.com",
  "error": "No data scraped from the endpoint. Check the URL or site settings."
}
```

**Internal error:**

```json
event: "run-scrape-result"
data: {
  "success": false,
  "url": "https://example-shop.com",
  "error": "Internal error in scraper pipeline",
  "details": "<exception text>"
}
```

### Output files

* Base output directory: `OUTPUT_DIR = "<cwd>/csv files"`

**Single collection mode (`url` contains `/collections/`):**

* Runs `scrape_shopify_products()` → `preprocess_variants_df()`
* Output: `<OUTPUT_DIR>/<collection_name>.csv`

**Site-root mode:**

* Runs `scrape_all_collections()` → `preprocess_variants_df()`
* Output:

  * Combined preprocessed CSV: `<OUTPUT_DIR>/<site_name>_combined.csv`
  * Per-collection raw CSVs: `<OUTPUT_DIR>/<site_name>/<collection_name>.csv`

The `<site_name>` / `<collection_name>` are derived from URLs via `url_to_name()`.

---

## 4. Weekly Scrape URL Configuration

### `set-weekly-url` (client → server)

**Description:** Set the URL scraped by the weekly scheduler.

**Request payload:**

```json
{
  "url": "https://example-shop.com"
}
```

* `url` (string, required): Must be a valid absolute URL with scheme + host.

**Success response:**

```json
event: "set-weekly-url-result"
data: {
  "success": true,
  "weekly_url": "https://example-shop.com",
  "day": "monday",
  "time": "00:00"
}
```

**Error responses:**

Missing URL:

```json
{
  "success": false,
  "error": "Missing 'url' in message"
}
```

Invalid URL:

```json
{
  "success": false,
  "error": "Invalid URL provided",
  "url": "not-a-url"
}
```

---

## 5. Weekly Scheduler

The scheduler is started at server boot:

```python
if __name__ == '__main__':
    start_scheduler()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

### Configuration

* `WEEKLY_DAY` (string): e.g. `"monday"`, `"tuesday"`, ...
* `WEEKLY_TIME` (string): e.g. `"00:00"` (24h format)

The scheduler sets up:

```python
schedule.every().monday.at("00:00").do(weekly_scrape_job)
```

(adjusting day/time based on configuration)

### Job behavior: `weekly_scrape_job`

On each scheduled run:

1. If `WEEKLY_SCRAPE_URL` is **not set**:

   ```json
   event: "weekly-scrape-result"
   data: {
     "success": false,
     "message": "WEEKLY_SCRAPE_URL is not set yet. Skipping.",
     "timestamp": "<ISO timestamp>"
   }
   ```

2. If `WEEKLY_SCRAPE_URL` is set, it calls `run_scrape_pipeline(WEEKLY_SCRAPE_URL)`.

   **On success:**

   ```json
   event: "weekly-scrape-result"
   data: {
     "success": true,
     "url": "https://example-shop.com",
     "file_name": "<csv file name>",
     "file_path": "<server path>",
     "rows": 1234,
     "timestamp": "<ISO timestamp>"
   }
   ```

   **On error:**

   ```json
   event: "weekly-scrape-result"
   data: {
     "success": false,
     "message": "Error during weekly scrape: <exception>",
     "url": "https://example-shop.com",
     "timestamp": "<ISO timestamp>"
   }
   ```

This event is broadcast to **all** connected clients.

---

## 6. Query API

### `query_with_inputs` (client → server)

**Description:** Main search / question-answering endpoint over product CSVs.

Supports two usage modes:

1. **Inline CSV content** (legacy / testing mode)
2. **URL-based** lookup, using previously scraped CSV files

#### 6.1 Request fields

```json
{
  "csv_text": "col1,col2\n...",             // OPTION 1: inline CSV
  "page_url": "https://example.com/...",   // OPTION 2A: URL
  "url": "https://example.com/...",        // OPTION 2B: alias of page_url
  "query": "What is the price of the hoodie?",
  "top_k": 3,
  "request_id": "abc123"
}
```

Accepted variants for `query`: `query`, `quer`, or `q`.

**Requirements:**

* A non-empty `query` (**required**).
* Exactly one CSV source:

  * either `csv_text` is provided and non-empty, OR
  * `page_url` / `url` is provided.

**Error: missing query**

```json
{
  "success": false,
  "error": "Missing 'query' (or 'quer'/'q') in message",
  "request_id": "abc123"
}
```

**Error: no CSV source provided**

```json
{
  "success": false,
  "error": "Provide either 'csv_text' or 'page_url'/'url' in the message",
  "request_id": "abc123"
}
```

#### 6.2 CSV source resolution

**Inline CSV mode (`csv_text`)**

* `df = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")`
* `csv_source = "inline"`, `csv_path_used = null`

**URL-based mode (`page_url` / `url`)**

1. `csv_path = resolve_csv_for_url(page_url, OUTPUT_DIR)`

   * Tries per-collection CSV: `OUTPUT_DIR/<site_name>/<collection_name>.csv`
   * Falls back to combined: `OUTPUT_DIR/<site_name>_combined.csv`

2. If no CSV found:

   ```json
   {
     "success": false,
     "error": "No CSV found for this URL. Run /run-scrape first for this site.",
     "page_url": "https://example.com/...",
     "request_id": "abc123"
   }
   ```

3. If CSV exists:

   * Load `raw_df = pd.read_csv(csv_path, dtype=str)`
   * If it **does not** contain `search_content` or `indexed_text_lemma`:

     * Treat as raw scraper output and run `preprocess_variants_df(raw_df)`
   * Else:

     * Use as preprocessed DataFrame
   * `df = df.fillna("")`
   * `csv_source = "file"`, `csv_path_used = csv_path`

#### 6.3 Follow-up logic

The server inspects the last turn in `SESSION_MEMORY[sid]["history"]` to support follow-up questions (e.g. “What is the price of this product?”).

* Extracts the last top product:

  * `previous_product_title`
  * `previous_sku`

* Uses `looks_like_followup(query, has_previous_product)`:

  * Checks for:

    * short questions like `"Price?"`
    * phrases like `"price of the product"`, `"discount on this product"`, `"is this product in stock"`
    * generic follow-up keywords with small word counts (<= 9 or <= 12)
    * references like `"above product"`, `"previous item"`, etc.

* If treated as follow-up and a previous product exists:

  ```text
  effective_query = "<user query> for product: <previous_product_title> (SKU: <previous_sku>)"
  ```

* If it “looks like” a new product name (longer query, no explicit “this/that/product/above/...”), the follow-up flag is disabled and `effective_query = query`.

This `effective_query` is what is actually passed to the retrieval engine.

#### 6.4 Retrieval & intent handling

Internally, `handle_query_with_uploaded_csv(df, effective_query, top_k)`:

* Chooses text field to index (`indexed_text_lemma` → `search_content` → fallback)

* Builds TF-IDF index with bigrams

* Computes cosine similarity

* If best score is below `SIMILARITY_THRESHOLD` (0.20) and no strong title match:

  * Returns a fallback message:

    ```text
    "Sorry, I couldn't answer that. I can assist you with product, website, business or item-related queries."
    ```

* Otherwise:

  * Examines top result and question intent via `detect_question_intent(query)`:

    Possible intents:

    * `discount`
    * `orig_price`
    * `price`
    * `stock`
    * `detail`
    * `general`

  * Uses appropriate columns:

    * `price_current`
    * `price_original`
    * `discount_percent`
    * `stock_status`
    * `summary`
    * `long_description`

  * Generates a natural language `final_answer` based on intent.

#### 6.5 Success response format

```json
event: "query_result"
data: {
  "success": true,
  "request_id": "abc123",
  "result": {
    "query": "What is the price of the hoodie?",
    "top_results": [
      {
        "clean_text": "...",
        "product_title": "Cool Hoodie (Color: Black)",
        "url": "https://shop.com/products/cool-hoodie?variant=123",
        "score": 0.87,
        "sku": "HD-123-BLK",
        "price_current": 999.0,
        "price_original": 1299.0,
        "discount_percent": 23.1,
        "stock_status": "In Stock",
        "summary": "Short summary of the product...",
        "long_description": "Full detailed description..."
      }
      // up to top_k entries
    ],
    "final_answer": "The current price of Cool Hoodie (Color: Black) is 999.0. It currently has a discount of 23.1%.",
    "product_links": [
      "https://shop.com/products/cool-hoodie?variant=123",
      "https://shop.com/products/another-hoodie?variant=456"
    ]
  },
  "meta": {
    "csv_source": "inline",          // "inline" or "file"
    "csv_path": null,                // CSV path when csv_source == "file"
    "page_url": "https://example..." // original page_url if provided
  }
}
```

#### 6.6 Error response format

```json
event: "query_result"
data: {
  "success": false,
  "request_id": "abc123",
  "error": "Exception during query handling",
  "details": "<exception text>"
}
```

---

## 7. Minimal Frontend Example (JavaScript)

```js
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Shopify Scraper + Query Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>

  <style>
    body {
      font-family: Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 1100px;
      margin: auto;
      padding: 20px;
    }
    .section {
      padding: 16px;
      background: #1e293b;
      border-radius: 10px;
      margin-bottom: 18px;
      border: 1px solid #334155;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    h1 {
      margin-bottom: 10px;
      color: #f8fafc;
    }
    label {
      font-size: 0.85rem;
      color: #94a3b8;
    }
    input, textarea {
      width: 100%;
      padding: 8px;
      border-radius: 8px;
      border: 1px solid #334155;
      background: #0f172a;
      color: #e2e8f0;
      margin-top: 4px;
      margin-bottom: 10px;
    }
    button {
      padding: 8px 16px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      color: white;
      margin-right: 6px;
    }
    .btn-connect { background: #10b981; }
    .btn-disconnect { background: #ef4444; }
    .btn-primary { background: #3b82f6; }
    .btn-secondary { background: #64748b; }

    .status-pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 10px;
      font-size: 0.8rem;
      margin-left: 10px;
    }
    .connected { background: #065f46; color: #a7f3d0; }
    .disconnected { background: #7f1d1d; color: #fecaca; }

    .log-box {
      max-height: 300px;
      overflow-y: auto;
      background: #0f172a;
      border: 1px solid #334155;
      padding: 10px;
      border-radius: 8px;
      font-size: 0.85rem;
      white-space: pre-wrap;
    }
    .log-entry {
      border-bottom: 1px dashed #475569;
      padding: 6px;
      margin-bottom: 6px;
    }
  </style>
</head>

<body>
<div class="container">

  <h1>Shopify Scraper + Query Console</h1>

  <!-- ------------------- CONNECT / DISCONNECT ------------------- -->
  <div class="section">
    <h3>WebSocket Connection</h3>

    <button id="btnConnect" class="btn-connect">Connect</button>
    <button id="btnDisconnect" class="btn-disconnect">Disconnect</button>

    <span id="wsStatus" class="status-pill disconnected">Disconnected</span>
  </div>

  <!-- ------------------- SCRAPE SECTION ------------------- -->
  <div class="section">
    <h3>Run Scrape</h3>

    <label>Site / Collection URL</label>
    <input id="scrapeUrl" placeholder="https://example.com/collections/hoodies"/>

    <button id="btnRunScrape" class="btn-primary">Run Scrape</button>
  </div>

  <!-- ------------------- WEEKLY SCRAPE ------------------- -->
  <div class="section">
    <h3>Set Weekly Scrape URL</h3>

    <label>Weekly URL</label>
    <input id="weeklyUrl" placeholder="https://example.com"/>

    <button id="btnSetWeekly" class="btn-secondary">Set Weekly URL</button>
  </div>

  <!-- ------------------- QUERY BY PAGE URL ------------------- -->
  <div class="section">
    <h3>Ask Question (URL-based)</h3>

    <label>Page URL</label>
    <input id="queryPageUrl" placeholder="https://example.com/collections/t-shirts"/>

    <label>Question</label>
    <textarea id="queryTextUrl" placeholder="e.g., What is the cheapest t-shirt?"></textarea>

    <button id="btnQueryUrl" class="btn-primary">Ask</button>
  </div>

  <!-- ------------------- QUERY USING CSV ------------------- -->
  <div class="section">
    <h3>Query Using CSV</h3>

    <label>Upload CSV</label>
    <input type="file" id="csvFile" />

    <label>Question</label>
    <textarea id="queryTextCsv" placeholder="What is the discount on item X?"></textarea>

    <button id="btnQueryCsv" class="btn-secondary">Ask Using CSV</button>
  </div>

  <!-- ------------------- LOG ------------------- -->
  <div class="section">
    <h3>Logs</h3>
    <button id="btnClearLog" class="btn-secondary" style="margin-bottom:10px;">Clear</button>
    <div id="log" class="log-box"></div>
  </div>

</div>

<script>
  let socket = null;

  const log = (msg, obj=null) => {
    const logEl = document.getElementById("log");
    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.textContent = msg + (obj ? ("\n" + JSON.stringify(obj, null, 2)) : "");
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
  };

  const setStatus = (connected) => {
    const el = document.getElementById("wsStatus");
    el.className = "status-pill " + (connected ? "connected" : "disconnected");
    el.textContent = connected ? "Connected" : "Disconnected";
  };

  // ------------------ Connect Button ------------------
  document.getElementById("btnConnect").addEventListener("click", () => {
    if (socket && socket.connected) return;

    socket = io("http://localhost:5000", { autoConnect: false });
    socket.connect();

    socket.on("connect", () => { setStatus(true); log("Connected"); });
    socket.on("disconnect", () => { setStatus(false); log("Disconnected"); });

    // **All backend events**
    socket.on("system", (data) => log("SYSTEM", data));
    socket.on("run-scrape-status", (d) => log("SCRAPE STATUS", d));
    socket.on("run-scrape-result", (d) => log("SCRAPE RESULT", d));
    socket.on("set-weekly-url-result", (d) => log("WEEKLY SET", d));
    socket.on("weekly-scrape-result", (d) => log("WEEKLY SCRAPE RESULT", d));
    socket.on("query_result", (d) => log("QUERY RESULT", d));

    log("Connecting…");
  });

  // ------------------ Disconnect Button ------------------
  document.getElementById("btnDisconnect").addEventListener("click", () => {
    if (socket) {
      socket.disconnect();
      setStatus(false);
      log("Manually disconnected");
    }
  });

  // ------------------ Run Scrape ------------------
  document.getElementById("btnRunScrape").addEventListener("click", () => {
    const url = document.getElementById("scrapeUrl").value.trim();
    if (!url) return alert("Enter URL");

    socket.emit("run-scrape", { url });
    log("Sent SCRAPE request: " + url);
  });

  // ------------------ Set Weekly URL ------------------
  document.getElementById("btnSetWeekly").addEventListener("click", () => {
    const url = document.getElementById("weeklyUrl").value.trim();
    if (!url) return alert("Enter weekly URL");

    socket.emit("set-weekly-url", { url });
    log("Sent weekly URL set");
  });

  // ------------------ Query by URL ------------------
  document.getElementById("btnQueryUrl").addEventListener("click", () => {
    const pageUrl = document.getElementById("queryPageUrl").value.trim();
    const question = document.getElementById("queryTextUrl").value.trim();

    if (!pageUrl || !question) return alert("Enter both URL and question");

    socket.emit("query_with_inputs", {
      page_url: pageUrl,
      query: question,
      request_id: "url-" + Date.now()
    });

    log("Sent query (URL): " + question);
  });

  // ------------------ Query using CSV ------------------
  document.getElementById("btnQueryCsv").addEventListener("click", () => {
    const file = document.getElementById("csvFile").files[0];
    const question = document.getElementById("queryTextCsv").value.trim();

    if (!file || !question) return alert("Please upload CSV & enter question");

    const reader = new FileReader();
    reader.onload = (e) => {
      socket.emit("query_with_inputs", {
        csv_text: e.target.result,
        query: question,
        request_id: "csv-" + Date.now()
      });

      log("Sent query (CSV): " + question);
    };
    reader.readAsText(file);
  });

  // ------------------ Clear Log ------------------
  document.getElementById("btnClearLog").addEventListener("click", () => {
    document.getElementById("log").innerHTML = "";
  });
</script>

</body>
</html>
```

---

## 8. Notes & Limitations

* **Session memory** is in-memory only; it resets when the process restarts.
* **Scheduler** runs in a background daemon thread; ensure the process stays alive.
* Files are stored on local disk under `csv files/`; use a shared volume if running multiple instances.
* This API assumes the input URLs are Shopify stores using `/collections/` and `/products.json` endpoints.
