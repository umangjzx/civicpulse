# CivicPulse - AI-Powered Community Complaint Management System

## Overview

CivicPulse is a modern, full-featured web application for managing civic complaints and grievances. It leverages Artificial Intelligence to automatically categorize, prioritize, and analyze complaints for efficient governance and community engagement.

### Key Features

✨ **AI-Powered Complaint Analysis**
- Automatic categorization using Machine Learning
- Priority prediction based on complaint content
- Spam detection to filter invalid submissions
- Similar complaint detection to identify patterns

📍 **Geospatial Intelligence**
- Interactive maps for complaint visualization
- Heat mapping to identify problem areas
- Location-based filtering and analysis
- GPS coordinates tracking

📊 **Advanced Analytics Dashboard**
- Real-time performance metrics
- Department-level SLA tracking
- Trend analysis and reporting
- Resolution time insights
- User satisfaction metrics

👥 **User Management**
- User registration and authentication
- Role-based access control (Admin/User)
- User profile management
- Activity tracking

🔔 **Communication & Collaboration**
- Status update notifications
- Comments and discussion threads
- Admin-to-user communication
- Email notifications (extensible)

## Technology Stack

### Backend
- **Framework:** Flask 2.3.3
- **Database:** SQLite
- **ML/NLP:** scikit-learn, NLTK
- **Data Processing:** Pandas, NumPy

### Frontend
- **UI Framework:** Bootstrap 5.3
- **Mapping:** Leaflet.js
- **Charts:** Chart.js
- **JavaScript:** Vanilla JS + Bootstrap utilities

### AI/ML
- **Classification:** Naive Bayes, Random Forest
- **Text Processing:** TF-IDF Vectorization, NLTK Lemmatization
- **Similarity Detection:** Cosine Similarity
- **Spam Detection:** Pattern matching with heuristics

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Navigate to project directory:**
```bash
cd civicpulse
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Initialize the database:**
```bash
python -c "from database import Database; Database()"
```

5. **Generate sample data (optional):**
```bash
python data_processor.py generate 50 200
```

6. **Run the application:**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Default Credentials

### Admin Account
- **Username:** admin
- **Password:** Admin123

### Demo User
Generate with sample data command above

## Project Structure

```
civicpulse/
├── app.py                   # Main Flask application
├── database.py             # Database models and initialization
├── models.py              # ORM-like classes (User, Complaint, etc.)
├── ai_engine.py          # AI/ML prediction engine
├── data_processor.py     # Data generation and export utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── static/
│   ├── css/
│   │   └── style.css   # Custom styles
│   ├── js/
│   │   └── main.js     # JavaScript utilities
│   └── uploads/        # User-uploaded files
├── templates/
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   ├── login.html                 # Login page
│   ├── register.html              # Registration page
│   ├── dashboard.html             # User dashboard
│   ├── submit_complaint.html      # Complaint submission
│   ├── complaint_detail.html      # Complaint detail view
│   ├── admin_dashboard.html       # Admin dashboard
│   ├── admin_complaints.html      # Complaint management
│   └── admin_analytics.html       # Analytics dashboard
└── models/              # Trained ML models (auto-generated)
```

## API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login
- `GET /logout` - User logout

### Complaints
- `GET /dashboard` - User complaints list
- `GET /complaint/new` - Complaint submission form
- `POST /complaint/new` - Submit new complaint
- `GET /complaint/<id>` - View complaint details
- `GET /complaint/<id>/upvote` - Upvote a complaint

### Admin
- `GET /admin/dashboard` - Admin overview
- `GET /admin/complaints` - All complaints management
- `POST /admin/complaint/<id>/update` - Update complaint status
- `GET /admin/analytics` - Analytics dashboard

### API (JSON)
- `GET /api/complaints/geojson` - GeoJSON for maps
- `GET /api/analytics/department-performance` - Department metrics
- `GET /api/analytics/stats` - General statistics

## Usage Guide

### For Users

1. **Register/Login**
   - Create account or login with credentials

2. **Submit Complaint**
   - Click "Submit Complaint"
   - Provide detailed information
   - Add location and optional image
   - Submit for processing

3. **Track Status**
   - View all complaints in dashboard
   - Check AI insights
   - Upvote complaints

### For Administrators

1. **Dashboard**
   - View key metrics
   - Monitor system health

2. **Manage Complaints**
   - View all complaints
   - Update status and priority
   - Filter and search

3. **Analytics**
   - View trend charts
   - Department performance
   - Map visualization

## Data Export

### Using Python CLI

```bash
# Generate sample data
python data_processor.py generate 50 200

# Export to CSV
python data_processor.py export-csv complaints.csv

# Export to Excel
python data_processor.py export-excel complaints.xlsx

# View statistics
python data_processor.py stats
```

## AI Features

### Complaint Categorization
- Automatic categorization using Naive Bayes
- 8 categories: Roads, Water, Electricity, Waste, Parks, Safety, Construction, Other
- Confidence scoring

### Priority Prediction
- Outputs: Low, Medium, High
- Based on content analysis

### Spam Detection
- Pattern matching heuristics
- Score: 0.0 - 1.0

### Similar Complaints
- TF-IDF based similarity matching
- Groups related issues

## Troubleshooting

### Database Issues
```bash
rm civicpulse.db
python -c "from database import Database; Database()"
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use
Change port in app.py line 380

## Security Notes

⚠️ **For Production Use:**
- Change default credentials
- Enable HTTPS with SSL
- Use strong secret key
- Validate all inputs
- Restrict file uploads
- Enable database backups

## Future Enhancements

🚀 **Planned Features**
- Email notifications
- SMS integration
- Mobile app
- Multi-language support
- Advanced reporting
- Budget tracking
- Social media integration
- Real-time updates

## License

MIT License - Open Source

## Support

For issues, questions, or suggestions:
- Check README files in each directory
- Review code comments
- Test with sample data first

---

**Made with ❤️ for better civic engagement**
