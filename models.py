from datetime import datetime
from database import Database

db = Database()

class User:
    @staticmethod
    def create(username, email, password, full_name, address=None, phone=None, is_admin=False):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password, full_name, address, phone, is_admin)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password, full_name, address, phone, is_admin))
            return cursor.lastrowid
    
    @staticmethod
    def get_by_username(username):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            return cursor.fetchone()
    
    @staticmethod
    def get_by_id(user_id):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            return cursor.fetchone()

class Complaint:
    @staticmethod
    def create(user_id, title, description, category, location, latitude=None, longitude=None, image_path=None):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO complaints 
                (user_id, title, description, category, location, latitude, longitude, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, title, description, category, location, latitude, longitude, image_path))
            return cursor.lastrowid
    
    @staticmethod
    def get_all():
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, u.username, u.full_name 
                FROM complaints c
                JOIN users u ON c.user_id = u.id
                ORDER BY c.created_at DESC
            ''')
            return cursor.fetchall()
    
    @staticmethod
    def get_by_id(complaint_id):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, u.username, u.full_name, u.email
                FROM complaints c
                JOIN users u ON c.user_id = u.id
                WHERE c.id = ?
            ''', (complaint_id,))
            return cursor.fetchone()
    
    @staticmethod
    def update_status(complaint_id, status, changed_by, notes=None):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE complaints 
                SET status = ?, updated_at = CURRENT_TIMESTAMP,
                    resolved_at = CASE WHEN ? = 'Resolved' THEN CURRENT_TIMESTAMP ELSE resolved_at END
                WHERE id = ?
            ''', (status, status, complaint_id))
            
            # Log status change
            cursor.execute('''
                INSERT INTO status_logs (complaint_id, old_status, new_status, changed_by, notes)
                SELECT ?, status, ?, ?, ? FROM complaints WHERE id = ?
            ''', (complaint_id, status, changed_by, notes, complaint_id))
            
            conn.commit()
    
    @staticmethod
    def get_by_category(category):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM complaints WHERE category = ?', (category,))
            return cursor.fetchall()
    
    @staticmethod
    def upvote(complaint_id, user_id):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO upvotes (complaint_id, user_id) VALUES (?, ?)', (complaint_id, user_id))
                cursor.execute('UPDATE complaints SET upvotes = upvotes + 1 WHERE id = ?', (complaint_id,))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

class AIPrediction:
    @staticmethod
    def save(complaint_id, predicted_category, predicted_priority, spam_score, similar_complaints, confidence_score):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ai_predictions 
                (complaint_id, predicted_category, predicted_priority, spam_score, similar_complaints, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (complaint_id, predicted_category, predicted_priority, spam_score, similar_complaints, confidence_score))
            return cursor.lastrowid

class Analytics:
    @staticmethod
    def get_complaints_by_category():
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM complaints 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            return cursor.fetchall()
    
    @staticmethod
    def get_complaints_by_status():
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM complaints 
                GROUP BY status
            ''')
            return cursor.fetchall()
    
    @staticmethod
    def get_complaints_over_time(days=30):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM complaints
                WHERE created_at >= DATE('now', ?)
                GROUP BY DATE(created_at)
                ORDER BY date
            ''', (f'-{days} days',))
            return cursor.fetchall()
    
    @staticmethod
    def get_average_resolution_time():
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT AVG(
                    JULIANDAY(resolved_at) - JULIANDAY(created_at)
                ) * 24 as avg_hours
                FROM complaints
                WHERE status = 'Resolved' AND resolved_at IS NOT NULL
            ''')
            result = cursor.fetchone()
            return result[0] if result[0] else 0
    
    @staticmethod
    def get_sla_breach_percentage():
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE 
                        WHEN JULIANDAY(COALESCE(resolved_at, 'now')) - JULIANDAY(created_at) * 24 > d.sla_hours 
                        THEN 1 ELSE 0 
                    END) as breaches
                FROM complaints c
                LEFT JOIN departments d ON c.department = d.name
                WHERE c.status = 'Resolved'
            ''')
            result = cursor.fetchone()
            if result[0] and result[0] > 0:
                return (result[1] / result[0]) * 100
            return 0