import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_name='civicpulse.db'):
        self.db_name = db_name
        self.init_db()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name)
    
    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    address TEXT,
                    phone TEXT,
                    is_admin BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Complaints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    location TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    image_path TEXT,
                    status TEXT DEFAULT 'Submitted',
                    priority TEXT DEFAULT 'Medium',
                    upvotes INTEGER DEFAULT 0,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Comments/Responses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    is_admin BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (complaint_id) REFERENCES complaints (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Status logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS status_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id INTEGER NOT NULL,
                    old_status TEXT,
                    new_status TEXT NOT NULL,
                    changed_by INTEGER NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (complaint_id) REFERENCES complaints (id),
                    FOREIGN KEY (changed_by) REFERENCES users (id)
                )
            ''')
            
            # AI Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id INTEGER NOT NULL,
                    predicted_category TEXT,
                    predicted_priority TEXT,
                    spam_score REAL,
                    similar_complaints TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (complaint_id) REFERENCES complaints (id)
                )
            ''')
            
            # Upvotes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS upvotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(complaint_id, user_id)
                )
            ''')
            
            # Notifications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    complaint_id INTEGER,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    type TEXT DEFAULT 'info',
                    is_read BOOLEAN DEFAULT 0,
                    action_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (complaint_id) REFERENCES complaints (id)
                )
            ''')
            
            # Departments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS departments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    sla_hours INTEGER DEFAULT 72
                )
            ''')
            
            # Insert default departments
            departments = [
                ('Roads & Transportation', 'roads@city.gov', '555-1001', 48),
                ('Water & Sewage', 'water@city.gov', '555-1002', 24),
                ('Electricity', 'power@city.gov', '555-1003', 12),
                ('Waste Management', 'waste@city.gov', '555-1004', 72),
                ('Parks & Recreation', 'parks@city.gov', '555-1005', 120),
                ('Public Safety', 'safety@city.gov', '555-1006', 6),
                ('Building & Construction', 'building@city.gov', '555-1007', 96)
            ]
            
            cursor.execute("SELECT COUNT(*) FROM departments")
            if cursor.fetchone()[0] == 0:
                cursor.executemany(
                    "INSERT INTO departments (name, email, phone, sla_hours) VALUES (?, ?, ?, ?)",
                    departments
                )
            
            # Create admin user (password: Admin123)
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, email, password, full_name, is_admin) VALUES (?, ?, ?, ?, ?)",
                ('admin', 'admin@civicpulse.gov', 'scrypt:32768:8:1$NxFuENVCdMgxjPLn$e67be61e11ff7c78ce03328c9435b3d550c1afba5d591091dc663dda3d8d985536720c55cda1a40b20c8b0e89993940d2fc96cd14bf367d77c332e8faff069c2', 'System Administrator', 1)
            )
            
            conn.commit()