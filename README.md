# EarthScape Climate Agency - Django MVC Architecture

A comprehensive Django-based climate data monitoring and analytics system implementing proper MVC (Model-View-Controller) architecture with advanced data processing capabilities.

## 🏗️ Architecture Overview

This project implements a clean MVC architecture where:

- **Models**: Django ORM models handle data persistence and business logic
- **Views**: Django views handle HTTP requests and responses
- **Controllers**: Custom controller classes contain all business logic and data processing
- **Templates**: Django templates provide the frontend UI

## 📊 Features

### Core Functionality
- **User Authentication & Authorization** with role-based access (Admin, Analyst, Viewer)
- **Climate Data Management** with support for multiple data sources
- **Real-time Data Processing** and analytics
- **Interactive Dashboard** with comprehensive metrics
- **Alert System** with configurable thresholds
- **Data Quality Monitoring** and integrity checks
- **Feedback & Support System**

### Data Sources Supported
- Satellite Imagery
- Weather Stations
- Environmental Sensors
- Historical Data
- Real-time Data Streams

### Environmental Metrics Tracked
- Temperature (°C)
- Humidity (%)
- Precipitation (mm)
- Wind Speed (m/s) & Direction
- Atmospheric Pressure (hPa)
- Air Quality (PM2.5, PM10, CO2 levels)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Django 5.2+
- SQLite (default) or PostgreSQL/MySQL

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd climateeproject
pip install -r requirements.txt
```

2. **Run migrations:**
```bash
python manage.py makemigrations
python manage.py migrate
```

3. **Create superuser:**
```bash
python manage.py createsuperuser
```

> **Note**: A test admin user has been created for you:
> - **Username**: `testadmin`
> - **Password**: `testpassword123`

4. **Run development server:**
```bash
python manage.py runserver
```

5. **Access the application:**
   - Main site: http://127.0.0.1:8000/
   - Admin panel: http://127.0.0.1:8000/admin/

## 🏛️ MVC Architecture Details

### Controllers

Each app contains dedicated controller classes that handle business logic:

#### `authentication/controllers.py`
- `AuthenticationController`: Handles user login, registration, logout, and profile management

#### `dashboard/controllers.py`
- `DashboardController`: Manages dashboard data aggregation, metrics calculation, and user-specific views

#### `data/controllers.py`
- `DataController`: Handles data ingestion, processing, analytics, and integrity validation

#### `notifications/controllers.py`
- `NotificationsController`: Manages alerts, notifications, and feedback systems

### Models Structure

```
authentication/
├── User (Custom user model with roles)

data/
├── DataSource (Climate data sources)
├── ClimateData (Environmental measurements)
└── DataUpload (File upload tracking)

dashboard/
├── DashboardWidget (Customizable widgets)
└── UserDashboard (User dashboard configs)

notifications/
├── AlertThreshold (Alert configuration)
├── Notification (System notifications)
└── Feedback (User feedback system)
```

## 🔧 Key Components

### Data Processing Pipeline
1. **Ingestion**: Multiple data source types with validation
2. **Processing**: Real-time data cleaning and quality scoring
3. **Storage**: Optimized database schema with indexes
4. **Analytics**: Advanced aggregation and trend analysis
5. **Visualization**: Interactive charts and dashboards

### Alert System
- Configurable thresholds for environmental metrics
- Real-time monitoring and notification generation
- Role-based alert distribution
- Historical alert tracking and analysis

### Quality Assurance
- Data integrity validation
- Quality scoring algorithms
- Completeness and accuracy metrics
- Automated data cleansing

## 📈 Analytics & Reporting

### Dashboard Metrics
- **Core Statistics**: Total data points, active sources, growth rates
- **Environmental Metrics**: Temperature, humidity, air quality trends
- **Data Quality**: Completeness, accuracy, integrity scores
- **System Health**: Performance, error rates, storage utilization
- **Source Performance**: Uptime, data volume, quality metrics

### Advanced Analytics
- Climate pattern recognition
- Correlation analysis
- Trend forecasting
- Anomaly detection
- Predictive modeling (framework ready)

## 🔐 Security Features

- Role-based access control (RBAC)
- Secure authentication with Django's auth system
- CSRF protection on all forms
- Data encryption for sensitive information
- Audit logging for data modifications

## 🗃️ Database Schema

The system uses Django's ORM with optimized queries:

```sql
-- Key relationships
User (1) -- (*) DataSource
DataSource (1) -- (*) ClimateData
DataSource (1) -- (*) DataUpload
AlertThreshold (1) -- (*) Notification
User (1) -- (*) Notification (recipients)
User (1) -- (*) Feedback
```

## 📊 API Endpoints

### Authentication
- `/auth/login/` - User login
- `/auth/register/` - User registration
- `/auth/logout/` - User logout
- `/auth/profile/` - Profile management

### Dashboard
- `/` - Main dashboard (redirects to dashboard/)
- `/dashboard/` - Comprehensive dashboard view

### Data Management
- `/data/sources/` - Data source management
- `/data/upload/` - Data file upload
- `/data/analytics/` - Data analytics view

### Notifications
- `/notifications/` - Notification center
- `/notifications/feedback/` - Feedback submission
- `/notifications/alerts/` - Alert management

## 🧪 Testing

Run tests with:
```bash
python manage.py test
```

## 📚 Documentation

### Data Format Specifications
- CSV: Standard climate data format
- JSON: API-compatible data structure
- XML: Legacy system compatibility

### Configuration
- Alert thresholds in `notifications/models.py`
- Dashboard widgets in `dashboard/controllers.py`
- Data validation rules in `data/controllers.py`

## 🤝 Contributing

1. Follow MVC architecture patterns
2. Add controllers for new business logic
3. Update models for data structure changes
4. Maintain test coverage
5. Document new features

## 📄 License

This project is part of the EarthScape Climate Agency system.

## 🆘 Support

For technical support or questions:
- Check the documentation
- Review controller implementations
- Examine model relationships
- Test with sample data

---

**Built with Django 5.2+ for the EarthScape Climate Agency - Monitoring our planet's vital signs.**
