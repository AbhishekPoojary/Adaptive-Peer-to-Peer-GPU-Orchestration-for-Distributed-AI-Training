import sqlite3
import os

DB_PATH = "gpu_orchestration.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Add the agent_connected column, defaulting to 1 (since existing nodes are likely connected if they were online)
        cursor.execute("ALTER TABLE gpu_nodes ADD COLUMN agent_connected INTEGER DEFAULT 1;")
        print("Successfully added 'agent_connected' column to gpu_nodes.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column 'agent_connected' already exists.")
        else:
            print(f"Error: {e}")
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    migrate()
