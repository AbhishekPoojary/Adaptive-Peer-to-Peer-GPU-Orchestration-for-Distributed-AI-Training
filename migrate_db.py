import sqlite3
conn = sqlite3.connect('gpu_orchestration.db')
cur = conn.cursor()
cur.execute("PRAGMA table_info(gpu_nodes)")
cols = [row[1] for row in cur.fetchall()]
print('Existing columns:', cols)
if 'api_key_hint' not in cols:
    cur.execute("ALTER TABLE gpu_nodes ADD COLUMN api_key_hint VARCHAR(8)")
    conn.commit()
    print('Added api_key_hint column OK')
else:
    print('Column already exists - no migration needed')
conn.close()
