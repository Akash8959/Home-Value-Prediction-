import sqlite3

db_path = "data/predictions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add new column if it does not already exist
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN image_filename TEXT;")
    print("Column added successfully!")
except Exception as e:
    print("Migration skipped:", e)

conn.commit()
conn.close()
