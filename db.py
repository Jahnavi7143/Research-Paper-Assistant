import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "app_data.db"   # the database file (created automatically)

# Create the tables if they don't exist yet
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Table for user accounts
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')

    # Table for saved chat messages (each row = one message)
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')

    conn.commit()
    conn.close()


# Register a new user. Returns (success True/False, message)
def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Scramble (hash) the password before storing it
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hashed))
        conn.commit()
        return True, "Account created successfully! Please log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


# Check login. Returns the user's id if correct, otherwise None
def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if row is None:
        return None   # no such user

    user_id, stored_hash = row
    # Compare the entered password against the stored scrambled one
    if bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return user_id
    return None   # wrong password


# Save one chat message for a user
def save_message(user_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (user_id, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# Load all past messages for a user (oldest first)
def get_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE user_id = ? ORDER BY id ASC",
              (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": ct} for r, ct in rows]

# Delete all chat history for a user
def clear_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()