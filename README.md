# Campus Lost & Found

An **AI-powered** web application that helps campus students match lost and found items using **image similarity** — no manual browsing needed.

---

## 🚀 How It Works

1. A student posts a **lost** item with a photo.
2. Another student posts a **found** item with a photo.
3. The app automatically computes **AI similarity scores** between every new item and all existing items of the opposite status.
4. The **Top 5 matches** are displayed ranked by similarity percentage.

### Image Matching Engine

- **Model**: MobileNetV2 (pre-trained on ImageNet, no fine-tuning required)
  - `include_top=False, pooling='avg'` → produces a **1280-dimensional** feature embedding per image
  - Input images are resized to **224×224** before inference
- **Similarity**: Cosine similarity (via `scikit-learn`) between embedding vectors
  - Score range: 0.0 (unrelated) → 1.0 (identical)
  - Top 5 candidates from the opposite status pool are returned

### Database

- **SQLite** (`lost_found.db`) — zero configuration, single file, created automatically on first run.
- Schema (`items` table):

  | Column        | Type    | Description                                  |
  |---------------|---------|----------------------------------------------|
  | id            | INTEGER | Primary key                                  |
  | title         | TEXT    | Item name / short description                |
  | description   | TEXT    | Longer description                           |
  | status        | TEXT    | `'lost'` or `'found'`                        |
  | image_path    | TEXT    | Relative path to saved image file            |
  | embedding     | TEXT    | 1280-dim vector serialised as JSON           |
  | contact_info  | TEXT    | Email / phone of the poster                  |
  | timestamp     | TEXT    | UTC ISO-8601 timestamp                       |

---

## 📦 Setup

### Prerequisites
- Python 3.9+
- `pip`

### Installation

```bash
# 1. Navigate to the project directory
cd campus-lost-found

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

> **Note**: The first run will download the MobileNetV2 weights (~14 MB) from the internet.
> Subsequent runs use the cached weights.

### Open in Browser

Navigate to → **http://localhost:5000**

---

## 📁 Project Structure

```
campus-lost-found/
├── app.py            # Flask routes & application entry point
├── models.py         # SQLite schema & query helpers
├── embedding.py      # MobileNetV2 feature extraction
├── matching.py       # Cosine similarity matching engine
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── lost_found.db     # SQLite DB (auto-created on first run)
├── static/
│   └── uploads/      # Uploaded item images (auto-created)
└── templates/
    ├── base.html     # Shared layout (Bootstrap 5, dark theme)
    ├── index.html    # Homepage — all items
    ├── upload.html   # Post new item form
    └── results.html  # Item + top-5 AI matches
```

---

## 🔧 Configuration

| Variable              | Location | Default        | Description                  |
|-----------------------|----------|----------------|------------------------------|
| `app.secret_key`      | app.py   | pre-set string | Change for production!       |
| `MAX_CONTENT_LENGTH`  | app.py   | 16 MB          | Max upload size              |
| `MAX_DISPLAY_SIZE`    | app.py   | 600×600        | Max saved image dimensions   |
| `PORT`                | app.py   | 5000           | Flask dev server port        |

---

## ⚠️ Production Notes

- Replace `app.run(debug=True)` with a proper WSGI server (e.g., `gunicorn`).
- Store `secret_key` in an environment variable.
- Consider migrating from SQLite to PostgreSQL for concurrent writes at scale.
"# campus-lost-found" 
