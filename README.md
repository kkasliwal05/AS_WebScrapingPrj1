# 🌐 AI-Based E-Commerce Web Scraping & Query System

## 📌 Project Overview

This project is an **AI-powered backend system** that performs automated **web scraping**, **data preprocessing**, and **intelligent product search using NLP**.

It extracts product data from Shopify-based e-commerce websites and allows users to query products using **natural language**.

The system transforms raw website data into a **searchable product intelligence engine**.

---

## 🎯 Problem Statement

Manual product search on e-commerce platforms is inefficient and time-consuming.

This project solves that by:

* Automating product data extraction
* Structuring raw data into meaningful datasets
* Enabling intelligent search using natural language

---

## 🚀 Solution Approach

The system follows a **pipeline architecture**:

1. Scrape product data from Shopify websites
2. Clean and preprocess the extracted data
3. Convert text into vector form using NLP
4. Match user queries using similarity search
5. Return the most relevant results

---

## 🚀 Key Features

### 🔹 Web Scraping Engine

* Uses `/products.json` endpoint
* Supports:

  * Full website scraping
  * Collection-based scraping

**Extracted Data:**

* Product Title
* Price (Current & Original)
* Discount
* SKU
* Stock Status
* Product URL

---

### 🔹 Data Preprocessing

* Cleans HTML content
* Removes unnecessary words
* Normalizes product attributes
* Applies NLP techniques (NLTK lemmatization)

---

### 🔹 Intelligent Query Engine

* Uses:

  * **TF-IDF Vectorization**
  * **Cosine Similarity**

**Supported Queries:**

* "Show me black t-shirt under 1000"
* "Price of this product"
* "Is this product in stock?"

---

### 🔹 API System (FastAPI)

📍 **Local API URL:**
👉 http://127.0.0.1:8000/docs

**Endpoints:**

* `POST /scrape` → Scrape product data
* `POST /query` → Query products
* `POST /set-weekly-url` → Set scheduler
* `GET /weekly-status` → Check scheduler

---

### 🔹 WebSocket Integration

* Built using **Flask-SocketIO**
* Enables real-time query interaction

---

### 🔹 Scheduler (Automation)

* Weekly scraping automation
* Keeps data updated automatically

---

## 🛠️ Tech Stack

* **Language:** Python
* **Backend:** FastAPI, Flask
* **Web Scraping:** Requests, BeautifulSoup
* **NLP:** NLTK
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Real-time:** Flask-SocketIO

---

## ▶️ How to Run Locally

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Server

```bash
uvicorn api:app --reload
```

### Step 3: Open API

👉 http://127.0.0.1:8000/docs

---

## 🧪 Testing & Execution

The application was successfully tested using **FastAPI Swagger UI**.

### 🔹 Scrape API Example

```json
{
  "url": "https://99wholesale.com/"
}
```

### 🔹 Query API Example

```json
{
  "session_id": "1",
  "page_url": "https://99wholesale.com/",
  "query": "black t shirt under 1000"
}
```

### 🔹 Sample Response

```json
{
  "product_title": "Black Cotton T-Shirt",
  "price_current": "₹799",
  "stock_status": "In Stock"
}
```

---

## 🔄 System Workflow

User Input → Scraper → Data Cleaning → NLP Processing → Query Matching → Response

---

## 📸 Screenshots

![Swagger UI](https://github.com/user-attachments/assets/4c977cdf-880f-454e-b2ca-431463a24a59)

![API Execution](https://github.com/user-attachments/assets/e3bbc0bc-de14-4947-aa83-7848bbb1e4cd)

![Query Response](https://github.com/user-attachments/assets/cdefff0c-a2c7-44cf-873d-d24ecce54287)

---

## ⚠️ Limitations

* Works primarily with Shopify-based websites
* Depends on `/products.json` availability

---

## 🔐 Ethical Considerations

* Follows responsible scraping practices
* Avoids excessive server load
* Intended for educational use

---

## 🌍 Deployment

* Initially deployed on VPS using:

  * Nginx (Reverse Proxy)
  * Gunicorn (Application Server)
* Currently tested locally

---

## 🙋‍♀️ Author

**Khushi Kasliwal**

**Pavan Dogga**

---

## 📬 Contact & Connect

📧 Email: khushikasliwal4@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/khushi-kasliwal-953692260/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
