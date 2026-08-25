import sqlite3

conn = sqlite3.connect('lost_found.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT id, image_path FROM items').fetchall()
for row in rows:
    old = row['image_path']
    if old:
        # Normalise backslashes to forward slashes
        normalised = old.replace('\\', '/')
        # Strip leading 'static/' prefix so value is 'uploads/...'
        if normalised.startswith('static/'):
            normalised = normalised[len('static/'):]
        rowid = row['id']
        print(f"id={rowid}  old={old!r}  ->  new={normalised!r}")
        conn.execute('UPDATE items SET image_path=? WHERE id=?', (normalised, rowid))
conn.commit()
conn.close()
print('Done - all paths fixed in DB')
