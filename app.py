"""
app.py — Main Flask application for Campus Lost & Found.

Routes:
  GET  /              Homepage — list all items as cards
  GET  /upload        Show upload form
  POST /upload        Process new item, extract embedding, redirect to results
  GET  /results/<id>  Show item details + top-5 AI matches
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

from models import init_db, insert_item, get_item_by_id, get_all_items, resolve_item, get_resolved_count
from embedding import extract_embedding
from matching import find_top_matches

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'campus_lost_found_secret_key_2024'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_DISPLAY_SIZE = (600, 600)   # resize uploaded images before saving

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialise the SQLite database on startup
init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Check that the file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_image(file) -> str:
    """
    Save the uploaded file to UPLOAD_FOLDER, resizing it to MAX_DISPLAY_SIZE
    while preserving aspect ratio.

    Returns the path relative to the 'static' folder (e.g. 'uploads/abc123.jpg')
    so it can be passed directly to url_for('static', filename=...) in templates.
    On Windows, os.path.join uses backslashes, so we normalise to forward slashes.
    """
    import uuid
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    # Absolute path on disk (for writing the file)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

    # Open, resize, and save with Pillow
    img = Image.open(file).convert('RGB')
    img.thumbnail(MAX_DISPLAY_SIZE, Image.LANCZOS)
    img.save(save_path, quality=85, optimize=True)

    # Return only the part after 'static/' with forward slashes, e.g. 'uploads/abc.jpg'
    # so templates can use: url_for('static', filename=item['image_path'])
    relative = os.path.relpath(save_path, 'static').replace('\\', '/')
    return relative, save_path  # (url-safe relative path, full disk path for embedding)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Homepage — display all active items as Bootstrap cards."""
    items = get_all_items()
    resolved_count = get_resolved_count()
    return render_template('index.html', items=items, resolved_count=resolved_count)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    GET:  Render the upload form.
    POST: Validate and save the uploaded item, extract its embedding,
          store in the DB, then redirect to the results page.
    """
    if request.method == 'POST':
        # --- Validate form fields ---
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        status = request.form.get('status', '').strip()
        contact_info = request.form.get('contact_info', '').strip()

        if not title:
            flash('Please enter a title.', 'danger')
            return redirect(url_for('upload'))
        if status not in ('lost', 'found'):
            flash('Please select a valid status (lost or found).', 'danger')
            return redirect(url_for('upload'))

        # --- Validate and save image ---
        file = request.files.get('image')
        if not file or file.filename == '':
            flash('Please upload an image.', 'danger')
            return redirect(url_for('upload'))
        if not allowed_file(file.filename):
            flash('Unsupported file type. Please upload a PNG, JPG, JPEG, GIF, or WEBP.', 'danger')
            return redirect(url_for('upload'))

        image_path, full_disk_path = save_uploaded_image(file)

        # --- Extract embedding (may take a few seconds on first run) ---
        try:
            embedding = extract_embedding(full_disk_path)
        except Exception as e:
            flash(f'Error extracting image features: {e}', 'danger')
            return redirect(url_for('upload'))

        # --- Persist to database ---
        item_id = insert_item(
            title=title,
            description=description,
            status=status,
            image_path=image_path,
            embedding=embedding,
            contact_info=contact_info
        )

        flash('Item posted successfully! Here are your AI-powered matches.', 'success')
        return redirect(url_for('results', item_id=item_id))

    # GET — show the upload form
    return render_template('upload.html')


@app.route('/results/<int:item_id>')
def results(item_id):
    """
    Display the posted item alongside its top-5 AI matches from the
    opposite status pool (lost → matches with found items, and vice-versa).
    """
    item = get_item_by_id(item_id)
    if not item:
        flash('Item not found.', 'danger')
        return redirect(url_for('index'))

    import json
    embedding = json.loads(item['embedding']) if item['embedding'] else None
    matches = []
    if embedding:
        matches = find_top_matches(embedding, item['status'], top_n=5)

    opposite = 'found' if item['status'] == 'lost' else 'lost'
    return render_template('results.html', item=item, matches=matches, opposite=opposite)

@app.route('/resolve/<int:item_id>', methods=['POST'])
def resolve(item_id):
    """
    Mark an item as resolved — owner got their item back, or the found item
    was successfully returned.  Resolved items are hidden from the listing
    and excluded from the AI matching pool to reduce duplicates.
    """
    item = get_item_by_id(item_id)
    if not item:
        flash('Item not found.', 'danger')
        return redirect(url_for('index'))

    success = resolve_item(item_id)
    if success:
        flash(
            f'"{item["title"]}" has been marked as resolved ✅ — it will no longer appear in listings.',
            'success'
        )
    else:
        flash('Could not mark item as resolved. Please try again.', 'danger')

    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
