"""
Dashboard Controllers - Handle business logic for dashboard operations
"""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.db.models import Count, Avg, Max, Min, Q
from django.utils import timezone
from datetime import timedelta
from data.models import ClimateData, DataSource
from notifications.models import Notification

class DashboardController:
    """Controller for dashboard-related operations"""

    @staticmethod
    def get_comprehensive_dashboard_data():
        """Get comprehensive dashboard data with all metrics"""
        now = timezone.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        last_30d = now - timedelta(days=30)

        # Core statistics
        core_stats = {
            'total_data_points': ClimateData.objects.count(),
            'active_sources': DataSource.objects.filter(is_active=True).count(),
            'recent_data_points': ClimateData.objects.filter(timestamp__gte=last_24h).count(),
            'data_growth_7d': DashboardController._calculate_growth_rate(7),
            'data_growth_30d': DashboardController._calculate_growth_rate(30),
        }

        # Environmental metrics
        env_metrics = DashboardController._get_environmental_metrics(last_24h, last_7d, last_30d)

        # Data quality metrics
        quality_metrics = DashboardController._get_data_quality_metrics()

        # Source performance
        source_performance = DashboardController._get_source_performance(last_24h)

        # Recent activity
        recent_activity = DashboardController._get_recent_activity()

        # System health
        system_health = DashboardController._get_system_health()

        # Alert summary
        alert_summary = DashboardController._get_alert_summary()

        return {
            'core_stats': core_stats,
            'env_metrics': env_metrics,
            'quality_metrics': quality_metrics,
            'source_performance': source_performance,
            'recent_activity': recent_activity,
            'system_health': system_health,
            'alert_summary': alert_summary,
        }

    @staticmethod
    def _calculate_growth_rate(days):
        """Calculate data growth rate over specified days"""
        now = timezone.now()
        current_period = ClimateData.objects.filter(
            timestamp__gte=now - timedelta(days=days)
        ).count()

        previous_period = ClimateData.objects.filter(
            timestamp__gte=now - timedelta(days=days*2),
            timestamp__lt=now - timedelta(days=days)
        ).count()

        if previous_period == 0:
            return 0 if current_period == 0 else 100

        growth_rate = ((current_period - previous_period) / previous_period) * 100
        return round(growth_rate, 2)

    @staticmethod
    def _get_environmental_metrics(last_24h, last_7d, last_30d):
        """Get environmental metrics across different time periods"""
        from django.db.models import Case, When, FloatField
        
        # Handle both temperature formats (temp_max and temperature)
        # Calculate average temperature from both fields
        temp_avg_max = ClimateData.objects.filter(
            temp_max__isnull=False
        ).aggregate(avg=Avg('temp_max'))['avg']
        
        temp_avg_standard = ClimateData.objects.filter(
            temperature__isnull=False
        ).aggregate(avg=Avg('temperature'))['avg']
        
        # Use temp_max if available, otherwise use temperature
        current_avg = temp_avg_max if temp_avg_max is not None else temp_avg_standard
        
        # For 24h min/max, check both fields
        temp_24h_max_max = ClimateData.objects.filter(
            timestamp__gte=last_24h, temp_max__isnull=False
        ).aggregate(min=Min('temp_max'), max=Max('temp_max'))
        
        temp_24h_standard = ClimateData.objects.filter(
            timestamp__gte=last_24h, temperature__isnull=False
        ).aggregate(min=Min('temperature'), max=Max('temperature'))
        
        min_24h = temp_24h_max_max['min'] if temp_24h_max_max['min'] is not None else temp_24h_standard['min']
        max_24h = temp_24h_max_max['max'] if temp_24h_max_max['max'] is not None else temp_24h_standard['max']
        
        # 7d average
        temp_7d_max = ClimateData.objects.filter(
            timestamp__gte=last_7d, temp_max__isnull=False
        ).aggregate(avg=Avg('temp_max'))['avg']
        
        temp_7d_standard = ClimateData.objects.filter(
            timestamp__gte=last_7d, temperature__isnull=False
        ).aggregate(avg=Avg('temperature'))['avg']
        
        avg_7d = temp_7d_max if temp_7d_max is not None else temp_7d_standard
        
        return {
            'temperature': {
                'current_avg': float(current_avg) if current_avg is not None else None,
                'min_24h': float(min_24h) if min_24h is not None else None,
                'max_24h': float(max_24h) if max_24h is not None else None,
                'avg_7d': float(avg_7d) if avg_7d is not None else None,
            },
            'humidity': {
                'current_avg': ClimateData.objects.filter(
                    humidity__isnull=False
                ).aggregate(avg=Avg('humidity'))['avg'],
                'avg_7d': ClimateData.objects.filter(
                    timestamp__gte=last_7d, humidity__isnull=False
                ).aggregate(avg=Avg('humidity'))['avg'],
            },
            'air_quality': {
                'avg_pm25': ClimateData.objects.filter(
                    pm25__isnull=False
                ).aggregate(avg=Avg('pm25'))['avg'],
                'avg_pm10': ClimateData.objects.filter(
                    pm10__isnull=False
                ).aggregate(avg=Avg('pm10'))['avg'],
                'avg_co2': ClimateData.objects.filter(
                    co2_level__isnull=False
                ).aggregate(avg=Avg('co2_level'))['avg'],
            },
            'weather': {
                'total_precipitation_24h': ClimateData.objects.filter(
                    timestamp__gte=last_24h, precipitation__gt=0
                ).aggregate(total=Count('id'))['total'],
                'avg_wind_speed': ClimateData.objects.filter(
                    wind_speed__isnull=False
                ).aggregate(avg=Avg('wind_speed'))['avg'],
            }
        }

    @staticmethod
    def _get_data_quality_metrics():
        """Get data quality metrics"""
        total_records = ClimateData.objects.count()
        if total_records == 0:
            return {
                'quality_score': 0,
                'high_quality_percentage': 0,
                'low_quality_percentage': 0,
                'complete_records_percentage': 0,
            }

        quality_records = ClimateData.objects.filter(quality_score__isnull=False).count()
        high_quality = ClimateData.objects.filter(quality_score__gte=0.8).count()
        low_quality = ClimateData.objects.filter(quality_score__lt=0.5).count()

        # Completeness check (records with all core metrics)
        complete_records = ClimateData.objects.filter(
            temperature__isnull=False,
            humidity__isnull=False,
            timestamp__isnull=False
        ).count()

        return {
            'quality_score': round((quality_records / total_records) * 100, 2),
            'high_quality_percentage': round((high_quality / total_records) * 100, 2),
            'low_quality_percentage': round((low_quality / total_records) * 100, 2),
            'complete_records_percentage': round((complete_records / total_records) * 100, 2),
        }

    @staticmethod
    def _get_source_performance(last_24h):
        """Get source performance metrics"""
        return DataSource.objects.filter(is_active=True).annotate(
            total_data_points=Count('climate_data'),
            recent_data_points=Count('climate_data', filter=Q(climate_data__timestamp__gte=last_24h)),
            avg_quality=Avg('climate_data__quality_score'),
            last_update=Max('climate_data__timestamp'),
            uptime_percentage=Count('climate_data', filter=Q(
                climate_data__timestamp__gte=timezone.now() - timedelta(days=1)
            )) / 24.0 * 100  # Simplified uptime calculation
        ).order_by('-recent_data_points')[:10]

    @staticmethod
    def _get_recent_activity():
        """Get recent system activity"""
        last_24h = timezone.now() - timedelta(hours=24)

        return {
            'latest_readings': ClimateData.objects.select_related('data_source').order_by('-timestamp')[:5],
            'new_sources': DataSource.objects.filter(created_at__gte=last_24h).count(),
            'data_processed': ClimateData.objects.filter(uploaded_at__gte=last_24h).count(),
            'alerts_generated': Notification.objects.filter(
                notification_type='ALERT',
                created_at__gte=last_24h
            ).count(),
        }

    @staticmethod
    def _get_system_health():
        """Get system health metrics"""
        # Data freshness check
        stale_threshold = timezone.now() - timedelta(hours=6)
        stale_sources = DataSource.objects.filter(
            is_active=True,
            climate_data__timestamp__lt=stale_threshold
        ).distinct().count()

        total_active_sources = DataSource.objects.filter(is_active=True).count()

        # Error rate (simplified - based on failed uploads)
        from data.models import DataUpload
        recent_uploads = DataUpload.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        )
        failed_uploads = recent_uploads.filter(status='FAILED').count()
        error_rate = (failed_uploads / max(recent_uploads.count(), 1)) * 100

        # Storage health (simplified)
        total_records = ClimateData.objects.count()
        storage_health = min(100, (1000000 - total_records) / 10000)  # Arbitrary threshold

        return {
            'data_freshness': max(0, 100 - (stale_sources / max(total_active_sources, 1)) * 100),
            'error_rate': round(error_rate, 2),
            'storage_health': round(storage_health, 2),
            'overall_health': round((100 + max(0, 100 - error_rate) + storage_health) / 3, 2),
        }

    @staticmethod
    def _get_alert_summary():
        """Get alert and notification summary"""
        now = timezone.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        return {
            'active_alerts': Notification.objects.filter(
                notification_type='ALERT',
                is_read=False
            ).count(),
            'alerts_24h': Notification.objects.filter(
                notification_type='ALERT',
                created_at__gte=last_24h
            ).count(),
            'alerts_7d': Notification.objects.filter(
                notification_type='ALERT',
                created_at__gte=last_7d
            ).count(),
            'unread_notifications': Notification.objects.filter(is_read=False).count(),
        }

    @staticmethod
    @login_required
    def dashboard_view(request):
        """Complete dashboard view - handles all dashboard logic"""
        data = DashboardController.get_comprehensive_dashboard_data()
        
        # Get data sources with statistics for the template
        data_sources = DataSource.objects.filter(is_active=True).annotate(
            data_count=Count('climate_data')
        ).order_by('-created_at')[:10]
        
        # Get latest temperature readings (handle both temp formats)
        latest_readings = ClimateData.objects.select_related('data_source').filter(
            Q(temperature__isnull=False) | Q(temp_max__isnull=False)
        ).order_by('-timestamp')[:10]
        
        # Process readings to get temperature value (handle both formats)
        latest_temperature = []
        for reading in latest_readings:
            temp_value = None
            if reading.temp_max is not None:
                temp_value = float(reading.temp_max)
            elif reading.temperature is not None:
                temp_value = float(reading.temperature)
            
            if temp_value is not None:
                # Create a simple object-like structure for template
                class TempReading:
                    def __init__(self, data_source, temperature, timestamp):
                        self.data_source = data_source
                        self.temperature = temperature
                        self.timestamp = timestamp
                
                latest_temperature.append(TempReading(
                    reading.data_source,
                    temp_value,
                    reading.timestamp
                ))

        # Add template-specific data
        
        # Prepare chart data from real temperature readings
        chart_labels = []
        chart_temperatures = []
        for reading in reversed(latest_temperature[:6]):  # Last 6 readings for chart
            chart_labels.append(reading.timestamp.strftime('%H:%M'))
            chart_temperatures.append(reading.temperature)
        
        # If not enough data, pad with defaults
        while len(chart_labels) < 6:
            chart_labels.insert(0, '--:--')
            chart_temperatures.insert(0, 0)
        
        import json
        
        data.update({
            # Map core_stats to stats for template compatibility
            'stats': data['core_stats'],
            'data_sources': data_sources,
            'latest_temperature': latest_temperature[:5],  # Show top 5
            # Chart data for JavaScript
            'chart_labels': json.dumps(chart_labels),
            'chart_temperatures': json.dumps(chart_temperatures),
            # Keep all other data
            'env_metrics': data['env_metrics'],
            'quality_metrics': data['quality_metrics'],
            'source_performance': data['source_performance'],
            'recent_activity': data['recent_activity'],
            'system_health': data['system_health'],
            'alert_summary': data['alert_summary'],
            # User-specific data
            'user_role': request.user.role,
            'user_name': request.user.get_full_name() or request.user.username,
            'user_notifications': Notification.objects.filter(
                recipients=request.user,
                is_read=False
            ).order_by('-created_at')[:3],
        })

        # Role-based customizations
        if request.user.role == 'VIEWER':
            # Limited data for viewers
            data['restricted_access'] = True
        elif request.user.role == 'ANALYST':
            # Additional analytics for analysts
            data['advanced_analytics'] = True
        elif request.user.role == 'ADMIN':
            # Full access for admins
            data['admin_access'] = True

        return render(request, 'dashboard/dashboard.html', data)
