"""
Data Controllers - Handle business logic for data operations
"""
import os
import csv
import json
import pandas as pd
from datetime import datetime
from django.db.models import Count, Avg, Max, Min, Q
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.core.files.storage import default_storage
from django.http import JsonResponse
from datetime import timedelta
from .models import DataSource, ClimateData, DataUpload
from ai_models.models import MLPrediction

class DataController:
    """Controller for data-related operations"""

    @staticmethod
    def get_data_sources_with_stats():
        """Get all data sources with comprehensive statistics"""
        return DataSource.objects.annotate(
            total_data_points=Count('climate_data'),
            recent_data_points=Count(
                'climate_data',
                filter=Q(
                    climate_data__timestamp__gte=timezone.now() - timedelta(hours=24)
                )
            ),
            last_update=Max('climate_data__timestamp'),
            avg_temperature=Avg('climate_data__temperature'),
            avg_humidity=Avg('climate_data__humidity')
        ).order_by('-last_update')

    @staticmethod
    def get_data_source_details(source_id):
        """Get detailed information about a specific data source"""
        source = DataSource.objects.get(id=source_id)
        data_stats = ClimateData.objects.filter(data_source=source).aggregate(
            total_points=Count('id'),
            avg_temp=Avg('temperature'),
            min_temp=Min('temperature'),
            max_temp=Max('temperature'),
            avg_humidity=Avg('humidity'),
            avg_co2=Avg('co2_level'),
            avg_pm25=Avg('pm25'),
            avg_pm10=Avg('pm10'),
            last_reading=Max('timestamp')
        )

        recent_readings = ClimateData.objects.filter(
            data_source=source
        ).order_by('-timestamp')[:20]

        return {
            'source': source,
            'stats': data_stats,
            'recent_readings': recent_readings,
        }

    @staticmethod
    def process_data_upload(upload_instance):
        """Process uploaded data file and create climate data records"""
        # This would contain the logic to parse different file formats
        # For now, it's a placeholder for the processing logic
        try:
            # Simulate processing
            records_created = 42  # Placeholder
            upload_instance.records_processed = records_created
            upload_instance.status = 'COMPLETED'
            upload_instance.completed_at = timezone.now()
            upload_instance.save()

            return True, records_created
        except Exception as e:
            upload_instance.status = 'FAILED'
            upload_instance.errors = str(e)
            upload_instance.save()
            return False, str(e)

    @staticmethod
    def get_comprehensive_analytics():
        """Get comprehensive analytics for the entire system - returns format expected by template"""
        now = timezone.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        last_30d = now - timedelta(days=30)

        # Get basic counts
        total_sources = DataSource.objects.filter(is_active=True).count()
        total_data_points = ClimateData.objects.count()

        # Calculate average temperature - use temp_max or temperature depending on what's available
        # Try temp_max first (for Seattle weather format), then temperature
        temp_avg = None
        
        # Get average from temp_max field (Seattle weather format)
        temp_max_avg = ClimateData.objects.filter(temp_max__isnull=False).aggregate(
            avg=Avg('temp_max')
        )['avg']
        
        # Get average from temperature field (standard format)
        temp_avg_standard = ClimateData.objects.filter(temperature__isnull=False).aggregate(
            avg=Avg('temperature')
        )['avg']
        
        # Use temp_max if available, otherwise use temperature
        if temp_max_avg is not None:
            temp_avg = float(temp_max_avg)
        elif temp_avg_standard is not None:
            temp_avg = float(temp_avg_standard)
        
        # If both exist, calculate weighted average (unlikely but possible)
        if temp_max_avg is not None and temp_avg_standard is not None:
            temp_max_count = ClimateData.objects.filter(temp_max__isnull=False).count()
            temp_standard_count = ClimateData.objects.filter(temperature__isnull=False).count()
            total_count = temp_max_count + temp_standard_count
            if total_count > 0:
                temp_avg = (float(temp_max_avg) * temp_max_count + float(temp_avg_standard) * temp_standard_count) / total_count

        # Get data by source - format expected by template
        data_by_source = DataSource.objects.filter(is_active=True).annotate(
            count=Count('climate_data')
        ).values('name', 'count').order_by('-count')

        # Convert to list format for template
        data_by_source_list = list(data_by_source)

        return {
            'total_sources': total_sources,
            'total_data_points': total_data_points,
            'avg_temperature': temp_avg,
            'data_by_source': data_by_source_list,
            # Additional stats for future use
            'data_points_24h': ClimateData.objects.filter(timestamp__gte=last_24h).count(),
            'data_points_7d': ClimateData.objects.filter(timestamp__gte=last_7d).count(),
            'data_points_30d': ClimateData.objects.filter(timestamp__gte=last_30d).count(),
        }

    @staticmethod
    def get_climate_patterns(time_range='7d'):
        """Analyze climate patterns and trends"""
        now = timezone.now()

        if time_range == '24h':
            start_date = now - timedelta(hours=24)
        elif time_range == '7d':
            start_date = now - timedelta(days=7)
        elif time_range == '30d':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=7)

        # Hourly aggregations for the selected period
        hourly_data = ClimateData.objects.filter(
            timestamp__gte=start_date
        ).extra(
            select={'hour': "strftime('%%Y-%%m-%%d %%H:00:00', timestamp)"}
        ).values('hour').annotate(
            avg_temp=Avg('temperature'),
            avg_humidity=Avg('humidity'),
            avg_pressure=Avg('atmospheric_pressure'),
            data_points=Count('id')
        ).order_by('hour')

        # Daily trends
        daily_trends = ClimateData.objects.filter(
            timestamp__gte=start_date
        ).extra(
            select={'date': "strftime('%%Y-%%m-%%d', timestamp)"}
        ).values('date').annotate(
            avg_temp=Avg('temperature'),
            min_temp=Min('temperature'),
            max_temp=Max('temperature'),
            avg_humidity=Avg('humidity'),
            total_precipitation=Count('precipitation', filter=Q(precipitation__gt=0))
        ).order_by('date')

        # Correlation analysis (simplified)
        correlations = {
            'temp_humidity': ClimateData.objects.filter(
                timestamp__gte=start_date,
                temperature__isnull=False,
                humidity__isnull=False
            ).count(),
            'temp_pressure': ClimateData.objects.filter(
                timestamp__gte=start_date,
                temperature__isnull=False,
                atmospheric_pressure__isnull=False
            ).count(),
        }

        return {
            'time_range': time_range,
            'hourly_data': list(hourly_data),
            'daily_trends': list(daily_trends),
            'correlations': correlations,
            'data_points': ClimateData.objects.filter(timestamp__gte=start_date).count(),
        }

    @staticmethod
    def validate_data_integrity():
        """Perform data integrity checks"""
        issues = []

        # Check for missing timestamps
        missing_timestamps = ClimateData.objects.filter(timestamp__isnull=True).count()
        if missing_timestamps > 0:
            issues.append(f"Found {missing_timestamps} records with missing timestamps")

        # Check for invalid coordinate ranges
        invalid_coords = DataSource.objects.filter(
            Q(latitude__lt=-90) | Q(latitude__gt=90) |
            Q(longitude__lt=-180) | Q(longitude__gt=180)
        ).count()
        if invalid_coords > 0:
            issues.append(f"Found {invalid_coords} data sources with invalid coordinates")

        # Check for data gaps (sources with no recent data)
        stale_sources = DataSource.objects.filter(
            is_active=True,
            climate_data__timestamp__lt=timezone.now() - timedelta(days=7)
        ).distinct().count()
        if stale_sources > 0:
            issues.append(f"Found {stale_sources} sources with no data in the last 7 days")

        # Check for duplicate timestamps per source
        duplicate_checks = []
        for source in DataSource.objects.filter(is_active=True):
            duplicates = ClimateData.objects.filter(data_source=source).values('timestamp').annotate(
                count=Count('id')
            ).filter(count__gt=1).count()
            if duplicates > 0:
                duplicate_checks.append(f"Source '{source.name}': {duplicates} duplicate timestamps")

        return {
            'total_issues': len(issues) + len(duplicate_checks),
            'issues': issues,
            'duplicate_issues': duplicate_checks,
            'integrity_score': max(0, 100 - (len(issues) + len(duplicate_checks)) * 10),
        }

    @staticmethod
    @login_required
    def data_sources_view(request):
        """Complete data sources management - Admin/Analyst only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        if request.method == 'POST':
            # Handle data source creation without forms
            name = request.POST.get('name')
            source_type = request.POST.get('source_type')
            location = request.POST.get('location', '')
            latitude = request.POST.get('latitude')
            longitude = request.POST.get('longitude')
            description = request.POST.get('description', '')

            try:
                source = DataSource.objects.create(
                    name=name,
                    source_type=source_type,
                    location=location,
                    latitude=latitude if latitude else None,
                    longitude=longitude if longitude else None,
                    description=description,
                    created_by=request.user
                )
                messages.success(request, f'Data source "{name}" created successfully!')
                return redirect('data:sources')
            except Exception as e:
                messages.error(request, f'Error creating data source: {str(e)}')

        # Get sources with statistics
        sources = DataController.get_data_sources_with_stats()

        context = {
            'sources': sources,
            'source_types': DataSource.SourceType.choices,
        }
        return render(request, 'data/sources.html', context)

    @staticmethod
    @login_required
    def data_upload_view(request):
        """Complete data upload processing - Analyst/Admin only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        if request.method == 'POST' and request.FILES.get('data_file'):
            uploaded_file = request.FILES['data_file']
            data_source_id = request.POST.get('data_source')

            try:
                # Validate data source
                if not data_source_id:
                    raise ValueError("Please select a data source")

                data_source = DataSource.objects.get(id=data_source_id, is_active=True)

                # Save file to media directory
                file_path = default_storage.save(
                    f'uploads/{uploaded_file.name}',
                    uploaded_file
                )

                # Create upload record
                upload_record = DataUpload.objects.create(
                    filename=uploaded_file.name,
                    file_path=file_path,
                    uploaded_by=request.user,
                    status='PROCESSING'
                )

                # Process the uploaded file
                records_processed = DataController.process_uploaded_file(upload_record, data_source)

                # Update upload record
                upload_record.records_processed = records_processed
                upload_record.status = 'COMPLETED'
                upload_record.completed_at = timezone.now()
                upload_record.save()

                messages.success(request,
                    f'File "{uploaded_file.name}" processed successfully! '
                    f'{records_processed} climate data records added.'
                )
                return redirect('data:upload')

            except Exception as e:
                # Mark upload as failed
                if 'upload_record' in locals():
                    upload_record.status = 'FAILED'
                    upload_record.errors = str(e)
                    upload_record.save()

                messages.error(request, f'Upload failed: {str(e)}')

        # Get recent uploads and active data sources
        recent_uploads = DataUpload.objects.select_related('uploaded_by').order_by('-created_at')[:10]
        active_sources = DataSource.objects.filter(is_active=True)

        context = {
            'recent_uploads': recent_uploads,
            'active_sources': active_sources,
        }
        return render(request, 'data/upload.html', context)

    @staticmethod
    @login_required
    def data_analytics_view(request):
        """Complete data analytics dashboard"""
        # Get comprehensive analytics
        analytics = DataController.get_comprehensive_analytics()

        # Add additional analytics for the template
        time_range = request.GET.get('range', '30d')

        # Get climate patterns for selected time range
        patterns = DataController.get_climate_patterns(time_range)

        # Data integrity check
        integrity = DataController.validate_data_integrity()

        context = {
            'analytics': analytics,
            'patterns': patterns,
            'integrity': integrity,
            'time_range': time_range,
            'range_options': [
                ('24h', 'Last 24 Hours'),
                ('7d', 'Last 7 Days'),
                ('30d', 'Last 30 Days')
            ]
        }
        return render(request, 'data/analytics.html', context)

    @staticmethod
    @login_required
    def detailed_analytics_view(request):
        """Detailed analytics page with trends, predictions, and anomalies graphs"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        # Get time range filter
        time_range = request.GET.get('range', '30d')
        now = timezone.now()
        
        if time_range == '24h':
            start_date = now - timedelta(hours=24)
        elif time_range == '7d':
            start_date = now - timedelta(days=7)
        elif time_range == '30d':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=30)

        # Get temperature predictions
        temperature_predictions = MLPrediction.objects.filter(
            prediction_type='TEMPERATURE_FORECAST',
            target_timestamp__gte=start_date
        ).order_by('target_timestamp')[:100]

        # Get anomaly predictions
        anomaly_predictions = MLPrediction.objects.filter(
            prediction_type='ANOMALY_SCORE',
            created_at__gte=start_date
        ).order_by('created_at')[:100]

        # Get actual climate data for comparison (handle both temp formats)
        climate_data = ClimateData.objects.filter(
            timestamp__gte=start_date
        ).order_by('timestamp')[:200]

        # Prepare data for charts
        # Temperature trends and predictions
        temp_timestamps = []
        temp_actual = []
        temp_predictions = []
        temp_prediction_timestamps = []
        
        for data in climate_data:
            temp_value = None
            if data.temp_max is not None:
                temp_value = float(data.temp_max)
            elif data.temperature is not None:
                temp_value = float(data.temperature)
            
            if temp_value is not None:
                temp_timestamps.append(data.timestamp.isoformat())
                temp_actual.append(temp_value)

        for pred in temperature_predictions:
            temp_prediction_timestamps.append(pred.target_timestamp.isoformat())
            temp_predictions.append(float(pred.prediction_value))

        # Anomaly scores
        anomaly_timestamps = []
        anomaly_scores = []
        anomaly_flags = []
        
        for pred in anomaly_predictions:
            anomaly_timestamps.append(pred.created_at.isoformat())
            anomaly_scores.append(float(pred.prediction_value))
            anomaly_flags.append(pred.is_anomaly)

        # Get summary statistics
        total_predictions = MLPrediction.objects.filter(created_at__gte=start_date).count()
        total_anomalies = MLPrediction.objects.filter(
            created_at__gte=start_date,
            is_anomaly=True
        ).count()
        avg_anomaly_score = MLPrediction.objects.filter(
            prediction_type='ANOMALY_SCORE',
            created_at__gte=start_date
        ).aggregate(avg=Avg('prediction_value'))['avg'] or 0

        context = {
            'time_range': time_range,
            'range_options': [
                ('24h', 'Last 24 Hours'),
                ('7d', 'Last 7 Days'),
                ('30d', 'Last 30 Days')
            ],
            'temp_timestamps': json.dumps(temp_timestamps),
            'temp_actual': json.dumps(temp_actual),
            'temp_prediction_timestamps': json.dumps(temp_prediction_timestamps),
            'temp_predictions': json.dumps(temp_predictions),
            'anomaly_timestamps': json.dumps(anomaly_timestamps),
            'anomaly_scores': json.dumps(anomaly_scores),
            'anomaly_flags': json.dumps(anomaly_flags),
            'total_predictions': total_predictions,
            'total_anomalies': total_anomalies,
            'avg_anomaly_score': round(avg_anomaly_score, 3),
            'has_data': len(temp_actual) > 0 or len(temp_predictions) > 0 or len(anomaly_scores) > 0
        }

        return render(request, 'data/detailed_analytics.html', context)

    @staticmethod
    def process_uploaded_file(upload_record, data_source):
        """Process uploaded climate data file and create database records"""
        file_path = upload_record.file_path.path
        records_created = 0

        try:
            # Determine file type and process accordingly
            if upload_record.filename.lower().endswith('.csv'):
                records_created = DataController._process_csv_file(file_path, data_source)
            elif upload_record.filename.lower().endswith('.json'):
                records_created = DataController._process_json_file(file_path, data_source)
            elif upload_record.filename.lower().endswith('.xml'):
                records_created = DataController._process_xml_file(file_path, data_source)
            else:
                raise ValueError(f"Unsupported file format: {upload_record.filename}")

        except Exception as e:
            upload_record.errors = str(e)
            upload_record.save()
            raise e

        return records_created

    @staticmethod
    def _process_csv_file(file_path, data_source):
        """Process CSV climate data file"""
        records_created = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            # Try to detect if header exists - look for various column names
            sample = file.read(1024)
            file.seek(0)
            sample_lower = sample.lower()
            has_header = ('temperature' in sample_lower or 'timestamp' in sample_lower or
                         'date' in sample_lower or 'precipitation' in sample_lower or
                         'temp_max' in sample_lower or 'temp_min' in sample_lower or
                         'wind' in sample_lower or 'weather' in sample_lower)

            reader = csv.DictReader(file) if has_header else csv.reader(file)

            for row_num, row in enumerate(reader, 1):
                try:
                    if has_header:
                        # Dict reader - handle multiple column name formats
                        timestamp_str = (row.get('timestamp') or row.get('date') or
                                       row.get('datetime') or row.get('DATE'))

                        # Handle temperature - could be single temp or max/min
                        temperature = (row.get('temperature') or row.get('temp') or
                                     row.get('temp_max') or row.get('TEMP_MAX'))
                        temp_max = row.get('temp_max') or row.get('TEMP_MAX')
                        temp_min = row.get('temp_min') or row.get('TEMP_MIN')

                        humidity = row.get('humidity') or row.get('rh') or row.get('HUMIDITY')
                        precipitation = (row.get('precipitation') or row.get('rain') or
                                       row.get('PRECIPITATION'))
                        wind_speed = (row.get('wind_speed') or row.get('wind') or
                                    row.get('WIND_SPEED') or row.get('WIND'))

                        # Weather description
                        weather_description = (row.get('weather') or row.get('weather_description') or
                                             row.get('WEATHER') or row.get('WEATHER_DESCRIPTION'))

                        # Air quality (if present)
                        co2_level = row.get('co2') or row.get('co2_level') or row.get('CO2')
                        pm25 = row.get('pm25') or row.get('pm2.5') or row.get('PM25')
                        pm10 = row.get('pm10') or row.get('PM10')

                    else:
                        # List reader - assume specific column order
                        # For Seattle weather format: date, precipitation, temp_max, temp_min, wind, weather
                        if len(row) < 2:
                            continue
                        timestamp_str = row[0]
                        precipitation = row[1] if len(row) > 1 else None
                        temp_max = row[2] if len(row) > 2 else None
                        temp_min = row[3] if len(row) > 3 else None
                        wind_speed = row[4] if len(row) > 4 else None
                        weather_description = row[5] if len(row) > 5 else None
                        temperature = temp_max  # Use temp_max as main temperature

                    # Parse timestamp
                    if timestamp_str:
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt)
                                if timestamp.year < 2000:  # Probably wrong century
                                    timestamp = timestamp.replace(year=timestamp.year + 2000)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format works, skip this row
                            continue
                    else:
                        # Use current time if no timestamp
                        timestamp = timezone.now()

                    # Create climate data record
                    ClimateData.objects.create(
                        data_source=data_source,
                        timestamp=timestamp,
                        temperature=float(temperature) if temperature and temperature != '' else None,
                        temp_max=float(temp_max) if temp_max and temp_max != '' else None,
                        temp_min=float(temp_min) if temp_min and temp_min != '' else None,
                        humidity=float(humidity) if humidity and humidity != '' else None,
                        precipitation=float(precipitation) if precipitation and precipitation != '' else None,
                        wind_speed=float(wind_speed) if wind_speed and wind_speed != '' else None,
                        weather_description=weather_description if weather_description and weather_description != '' else '',
                        co2_level=float(co2_level) if co2_level and co2_level != '' else None,
                        pm25=float(pm25) if pm25 and pm25 != '' else None,
                        pm10=float(pm10) if pm10 and pm10 != '' else None,
                        quality_score=0.85  # Default quality score
                    )

                    records_created += 1

                except (ValueError, TypeError) as e:
                    # Skip invalid rows but continue processing
                    continue
                except Exception as e:
                    # Log error but continue
                    continue

        return records_created

    @staticmethod
    def _process_json_file(file_path, data_source):
        """Process JSON climate data file"""
        records_created = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)

                # Handle different JSON structures
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and 'records' in data:
                    records = data['records']
                elif isinstance(data, dict) and 'data' in data:
                    records = data['data']
                else:
                    # Single record
                    records = [data]

                for record in records:
                    try:
                        timestamp_str = record.get('timestamp') or record.get('date') or record.get('datetime')

                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = timezone.now()

                        ClimateData.objects.create(
                            data_source=data_source,
                            timestamp=timestamp,
                            temperature=record.get('temperature'),
                            humidity=record.get('humidity'),
                            precipitation=record.get('precipitation'),
                            wind_speed=record.get('wind_speed'),
                            co2_level=record.get('co2_level'),
                            pm25=record.get('pm25'),
                            pm10=record.get('pm10'),
                            quality_score=record.get('quality_score', 0.85)
                        )

                        records_created += 1

                    except Exception:
                        continue

            except json.JSONDecodeError:
                raise ValueError("Invalid JSON file format")

        return records_created

    @staticmethod
    def _process_xml_file(file_path, data_source):
        """Process XML climate data file"""
        # Basic XML processing - can be enhanced based on specific XML schema
        records_created = 0

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Look for climate data elements (customize based on XML structure)
            for record_elem in root.findall('.//record') or root.findall('.//data'):
                try:
                    timestamp_str = record_elem.findtext('timestamp') or record_elem.findtext('date')

                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timezone.now()

                    ClimateData.objects.create(
                        data_source=data_source,
                        timestamp=timestamp,
                        temperature=record_elem.findtext('temperature'),
                        humidity=record_elem.findtext('humidity'),
                        precipitation=record_elem.findtext('precipitation'),
                        wind_speed=record_elem.findtext('wind_speed'),
                        co2_level=record_elem.findtext('co2_level'),
                        pm25=record_elem.findtext('pm25'),
                        pm10=record_elem.findtext('pm10'),
                        quality_score=0.85
                    )

                    records_created += 1

                except Exception:
                    continue

        except Exception as e:
            raise ValueError(f"XML processing error: {str(e)}")

        return records_created

    @staticmethod
    @login_required
    def delete_data_source(request, source_id):
        """Delete a data source - Admin/Analyst only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)

        if request.method == 'POST':
            try:
                source = get_object_or_404(DataSource, id=source_id)
                source_name = source.name
                source.delete()
                return JsonResponse({'success': True, 'message': f'Data source "{source_name}" deleted successfully'})
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)

        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

    @staticmethod
    @login_required
    def update_data_source(request, source_id):
        """Update a data source - Admin/Analyst only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)

        if request.method == 'POST':
            try:
                source = get_object_or_404(DataSource, id=source_id)
                
                # Update fields
                name = request.POST.get('name')
                source_type = request.POST.get('source_type')
                location = request.POST.get('location', '')
                latitude = request.POST.get('latitude')
                longitude = request.POST.get('longitude')
                description = request.POST.get('description', '')

                if name:
                    source.name = name
                if source_type:
                    source.source_type = source_type
                source.location = location
                try:
                    source.latitude = float(latitude) if latitude and latitude.strip() else None
                except (ValueError, TypeError):
                    source.latitude = None
                try:
                    source.longitude = float(longitude) if longitude and longitude.strip() else None
                except (ValueError, TypeError):
                    source.longitude = None
                source.description = description
                
                # Handle is_active checkbox
                is_active = request.POST.get('is_active', 'false')
                source.is_active = (is_active == 'on' or is_active == 'true' or is_active == 'True')
                
                source.save()
                
                return JsonResponse({
                    'success': True,
                    'message': f'Data source "{source.name}" updated successfully',
                    'source': {
                        'id': source.id,
                        'name': source.name,
                        'source_type': source.source_type,
                        'location': source.location,
                        'is_active': source.is_active
                    }
                })
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)

        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

    @staticmethod
    @login_required
    def toggle_data_source_active(request, source_id):
        """Toggle active status of a data source - Admin/Analyst only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)

        if request.method == 'POST':
            try:
                source = get_object_or_404(DataSource, id=source_id)
                source.is_active = not source.is_active
                source.save()
                
                status_text = 'activated' if source.is_active else 'deactivated'
                return JsonResponse({
                    'success': True,
                    'message': f'Data source "{source.name}" {status_text} successfully',
                    'is_active': source.is_active
                })
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)

        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

    @staticmethod
    @login_required
    def get_data_source(request, source_id):
        """Get data source details for editing - Admin/Analyst only"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)

        try:
            source = get_object_or_404(DataSource, id=source_id)
            return JsonResponse({
                'success': True,
                'source': {
                    'id': source.id,
                    'name': source.name,
                    'source_type': source.source_type,
                    'location': source.location or '',
                    'latitude': str(source.latitude) if source.latitude else '',
                    'longitude': str(source.longitude) if source.longitude else '',
                    'description': source.description or '',
                    'is_active': source.is_active
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
