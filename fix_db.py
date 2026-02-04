import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

c.execute("ALTER TABLE predictions ADD COLUMN satellite_image TEXT")

conn.commit()
conn.close()

print("Column added!")
