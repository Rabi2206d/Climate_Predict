"""
AI Models Controllers - Handle ML model loading, predictions, and analytics
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .models import MLModel, MLPrediction, MLModelMetrics
from data.models import ClimateData, DataSource


class AIModelController:
    """Controller for AI/ML model operations"""

    # Class variables to cache loaded models
    _loaded_models = {}

    @staticmethod
    def load_model(model_type):
        """Load trained ML model from file"""
        if model_type in AIModelController._loaded_models:
            return AIModelController._loaded_models[model_type]

        model_paths = {
            'anomaly': os.path.join(settings.BASE_DIR, 'MLModels', 'anomaly_model.pkl'),
            'temperature': os.path.join(settings.BASE_DIR, 'MLModels', 'temperature_model.pkl')
        }

        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}")

        model_path = model_paths[model_type]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                # Load model (pickle.load doesn't need encoding parameter in Python 3)
                try:
                    model_data = pickle.load(f)
                except Exception as e2:
                    # If loading fails, suggest retraining
                    raise Exception(f"Model file incompatible. Try retraining the model with current numpy version. Original error: {str(e2)}")

            # Handle different model formats
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                scaler = model_data.get('scaler')
            else:
                model = model_data
                scaler = None

            loaded_model = {
                'model': model,
                'scaler': scaler,
                'loaded_at': timezone.now()
            }

            AIModelController._loaded_models[model_type] = loaded_model
            return loaded_model
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Error unpickling model: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            # Check if it's a numpy compatibility issue
            if 'numpy._core' in str(e) or 'numpy' in str(e).lower():
                raise Exception(f"NumPy compatibility issue. The model was likely created with a different NumPy version. Please retrain the model or downgrade NumPy to match the training environment. Error: {str(e)}")
            raise Exception(f"Error loading {model_type} model: {str(e)}")

    @staticmethod
    def detect_anomalies(temperature_data, humidity_data=None, threshold=0.1):
        """Detect climate anomalies using trained Isolation Forest model"""
        try:
            # Try to load anomaly detection model, fallback to creating one if it fails
            try:
                model_data = AIModelController.load_model('anomaly')
                model = model_data['model']
                scaler = model_data['scaler']
                use_trained_model = True
            except (FileNotFoundError, Exception) as e:
                # Fallback: Create a simple Isolation Forest model on the fly
                print(f"Warning: Could not load trained model, creating fallback model: {str(e)}")
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=0.1, random_state=42)
                scaler = StandardScaler()
                
                # Train the fallback model with available data
                if isinstance(temperature_data, (list, np.ndarray)):
                    temp_array = np.array(temperature_data).reshape(-1, 1)
                    if humidity_data is not None and len(humidity_data) > 0:
                        hum_array = np.array(humidity_data).reshape(-1, 1)
                        # Only combine if lengths match
                        if len(temp_array) == len(hum_array):
                            features = np.hstack([temp_array, hum_array])
                        else:
                            # Lengths don't match, use temperature only
                            features = temp_array
                    else:
                        features = temp_array
                    
                    # Scale and fit the fallback model
                    features_scaled = scaler.fit_transform(features)
                    model.fit(features_scaled)
                else:
                    # For single values, create a minimal model
                    temp_value = float(temperature_data)
                    features = np.array([[temp_value]])
                    features_scaled = scaler.fit_transform(features)
                    model.fit(features_scaled)
                
                use_trained_model = False

            # Prepare data for prediction
            if isinstance(temperature_data, (list, np.ndarray)):
                temp_array = np.array(temperature_data).reshape(-1, 1)

                # Add humidity if available and has matching length
                if humidity_data is not None and len(humidity_data) > 0:
                    hum_array = np.array(humidity_data).reshape(-1, 1)
                    # Only combine if lengths match
                    if len(temp_array) == len(hum_array):
                        features = np.hstack([temp_array, hum_array])
                    else:
                        # Lengths don't match, use temperature only
                        features = temp_array
                else:
                    features = temp_array

                # Scale features if scaler was used during training
                if scaler:
                    features = scaler.transform(features)

                # Get anomaly scores (-1 for outliers, 1 for inliers)
                scores = model.decision_function(features)
                predictions = model.predict(features)

                # Convert to anomaly probability (higher score = more anomalous)
                anomaly_scores = -scores  # Make positive scores indicate anomalies

                return {
                    'anomaly_scores': anomaly_scores.tolist(),
                    'is_anomaly': (anomaly_scores > threshold).tolist(),
                    'predictions': predictions.tolist(),
                    'threshold': threshold
                }
            else:
                # Single value prediction
                temp_value = float(temperature_data)
                features = np.array([[temp_value]])

                if scaler:
                    features = scaler.transform(features)

                score = model.decision_function(features)[0]
                prediction = model.predict(features)[0]

                anomaly_score = -score
                is_anomaly = anomaly_score > threshold

                return {
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': bool(is_anomaly),
                    'prediction': int(prediction),
                    'threshold': threshold
                }

        except Exception as e:
            raise Exception(f"Anomaly detection failed: {str(e)}")

    @staticmethod
    def predict_temperature(features_data):
        """Predict temperature using trained regression model - requires trained model"""
        try:
            # Load temperature prediction model - required, no fallback
            model_data = AIModelController.load_model('temperature')
            model = model_data['model']
            scaler = model_data['scaler']

            # Prepare features for prediction
            if isinstance(features_data, dict):
                # Convert dict to array (assuming specific feature order)
                # This should match the training feature order: [humidity, pressure, wind_speed, hour_of_day, month, prev_temp]
                # If prev_temp is not provided, use the current/last temperature or a default
                prev_temp = features_data.get('prev_temp')
                if prev_temp is None:
                    # Try to get from recent temperature or use default
                    prev_temp = features_data.get('temperature', features_data.get('current_temp', 20.0))
                
                features = np.array([[
                    features_data.get('humidity', 50.0),
                    features_data.get('pressure', 1013.0),
                    features_data.get('wind_speed', 5.0),
                    features_data.get('hour_of_day', 12),
                    features_data.get('month', 6),
                    float(prev_temp)
                ]])
            elif isinstance(features_data, (list, np.ndarray)):
                features = np.array(features_data).reshape(1, -1)
            else:
                raise ValueError("Invalid features format")

            # Scale features if scaler exists
            if scaler:
                features = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features)[0]

            # Calculate confidence interval (simplified)
            confidence_range = {
                'lower': float(prediction - 2.0),  # Rough estimate
                'upper': float(prediction + 2.0)
            }

            return {
                'predicted_temperature': float(prediction),
                'confidence_range': confidence_range,
                'features_used': features.tolist()
            }

        except FileNotFoundError as e:
            raise Exception(f"Temperature prediction model not found. Please train the model first. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Temperature prediction failed: {str(e)}")

    @staticmethod
    @login_required
    def ai_analysis_view(request):
        """Main AI analysis view"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        # Get recent ML predictions
        recent_predictions = MLPrediction.objects.select_related('model').order_by('-created_at')[:20]
        anomaly_predictions = MLPrediction.objects.select_related('model').filter(prediction_type='ANOMALY_SCORE').order_by('-created_at')[:20]
        temperature_predictions = MLPrediction.objects.select_related('model').filter(prediction_type='TEMPERATURE_FORECAST').order_by('-created_at')[:20]

        # Add confidence percentage to predictions
        for prediction in recent_predictions:
            if prediction.confidence_score:
                prediction.confidence_percentage = int(prediction.confidence_score * 100)
            else:
                prediction.confidence_percentage = 0

        for prediction in anomaly_predictions:
            if prediction.confidence_score:
                prediction.confidence_percentage = int(prediction.confidence_score * 100)
            else:
                prediction.confidence_percentage = 0

        for prediction in temperature_predictions:
            if prediction.confidence_score:
                prediction.confidence_percentage = int(prediction.confidence_score * 100)
            else:
                prediction.confidence_percentage = 0

        # Get model status
        # Map choice values to both possible database formats
        # Django TextChoices can store either the value or attribute name
        model_type_mapping = {
            'ANOMALY': ['ANOMALY_DETECTION', 'ANOMALY'],  # Try both formats
            'TEMPERATURE': ['TEMPERATURE_FORECAST', 'TEMPERATURE'],
            # Exclude CLIMATE and AIR_QUALITY models
        }
        
        # Filter to only show ANOMALY and TEMPERATURE models
        allowed_model_types = ['ANOMALY', 'TEMPERATURE']
        
        models_status = []
        for choice_value, display_name in MLModel.ModelType.choices:
            # Skip CLIMATE and AIR_QUALITY models
            if choice_value not in allowed_model_types:
                continue
            # Try multiple possible database formats
            possible_types = model_type_mapping.get(choice_value, [choice_value])
            
            model = None
            for db_type in possible_types:
                model = MLModel.objects.filter(model_type=db_type, is_active=True).first()
                if model:
                    break
            
            if model:
                # Get accuracy - try model's stored score, or calculate from predictions
                accuracy = model.accuracy_score
                if accuracy is None:
                    # Calculate average confidence from recent predictions as accuracy proxy
                    predictions = MLPrediction.objects.filter(model=model)
                    if predictions.exists():
                        confidences = [p.confidence_score for p in predictions if p.confidence_score is not None]
                        if confidences:
                            accuracy = sum(confidences) / len(confidences)
                
                status = {
                    'name': display_name,
                    'status': model.get_status_display(),
                    'last_trained': model.last_trained,
                    'accuracy': accuracy
                }
            else:
                status = {
                    'name': display_name,
                    'status': 'Not Available',
                    'last_trained': None,
                    'accuracy': None
                }
            models_status.append(status)

        context = {
            'recent_predictions': recent_predictions,
            'anomaly_predictions': anomaly_predictions,
            'temperature_predictions': temperature_predictions,
            'models_status': models_status,
            'can_run_analysis': request.user.role in ['ADMIN', 'ANALYST'],
            'data_sources': DataSource.objects.filter(is_active=True).order_by('name')
        }

        return render(request, 'ai_models/dashboard.html', context)

    @staticmethod
    @login_required
    def run_climate_analysis(request):
        """Run comprehensive climate analysis using ML models"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        try:
            # Get data source ID from request (if provided)
            data_source_id = request.GET.get('data_source_id') or request.POST.get('data_source_id')
            data_source = None
            
            if data_source_id:
                try:
                    data_source = DataSource.objects.get(id=data_source_id, is_active=True)
                except DataSource.DoesNotExist:
                    messages.warning(request, f'Selected data source not found or inactive.')
                    return redirect('ai_models:ai_dashboard')
            
            # Get climate data for analysis - filter by data source if provided
            base_query = ClimateData.objects.all()
            
            if data_source:
                base_query = base_query.filter(data_source=data_source)
            
            # Try recent data first (last 30 days), then fallback to all available data
            recent_data = base_query.filter(
                timestamp__gte=timezone.now() - timedelta(days=30)
            ).order_by('-timestamp')[:100]
            
            # If no recent data, use all available data (for historical datasets like Seattle weather)
            if not recent_data.exists():
                recent_data = base_query.order_by('-timestamp')[:100]
            
            if not recent_data.exists():
                if data_source:
                    messages.warning(request, f'No climate data available for "{data_source.name}". Please upload data first.')
                else:
                    messages.warning(request, 'No climate data available for analysis. Please upload data first.')
                return redirect('ai_models:ai_dashboard')

            # Extract temperature and humidity data - handle both formats
            temperatures = []
            humidities = []
            
            for data in recent_data:
                # Handle both temperature formats: temp_max (Seattle format) or temperature (standard)
                temp_value = None
                if data.temp_max is not None:
                    temp_value = float(data.temp_max)
                elif data.temperature is not None:
                    temp_value = float(data.temperature)
                
                if temp_value is not None:
                    temperatures.append(temp_value)
                    # Add humidity for same index to keep lengths matching
                    if data.humidity is not None:
                        humidities.append(float(data.humidity))
                    else:
                        # Add None placeholder to keep indices aligned
                        humidities.append(None)
            
            # Need at least some temperature data
            if not temperatures:
                messages.warning(request, 'No temperature data found in the dataset. Please ensure your data includes temperature information.')
                return redirect('ai_models:ai_dashboard')

            # Run anomaly detection with error handling
            # Filter out None values from humidities to get valid humidity data
            valid_humidities = [h for h in humidities if h is not None] if humidities else None
            
            try:
                # Only pass humidities if we have valid data matching temperature length
                if valid_humidities and len(valid_humidities) == len(temperatures):
                    anomaly_results = AIModelController.detect_anomalies(temperatures, valid_humidities)
                else:
                    # Use temperature only if humidity data doesn't match
                    anomaly_results = AIModelController.detect_anomalies(temperatures, None)
            except Exception as e:
                messages.warning(request, f'Anomaly detection model issue: {str(e)}. Using statistical method instead.')
                # Fallback to simple statistical anomaly detection
                if temperatures:
                    temp_mean = np.mean(temperatures)
                    temp_std = np.std(temperatures)
                    threshold_temp = temp_mean + 2 * temp_std  # 2 standard deviations
                    
                    anomaly_scores = []
                    is_anomaly = []
                    for temp in temperatures:
                        # Calculate z-score
                        z_score = abs((temp - temp_mean) / temp_std) if temp_std > 0 else 0
                        anomaly_score = z_score / 3.0  # Normalize to 0-1 range
                        anomaly_scores.append(float(anomaly_score))
                        is_anomaly.append(z_score > 2)  # Anomaly if > 2 std devs
                    
                    anomaly_results = {
                        'anomaly_scores': anomaly_scores,
                        'is_anomaly': is_anomaly,
                        'threshold': 0.67  # Equivalent to 2 std devs
                    }
                else:
                    anomaly_results = {
                        'anomaly_scores': [],
                        'is_anomaly': [],
                        'threshold': 0.1
                    }

            # Generate predictions for next few hours
            predictions = []
            base_time = timezone.now()

            try:
                for i in range(6):  # Next 6 hours
                    prediction_time = base_time + timedelta(hours=i+1)

                    # Use average recent conditions for prediction
                    # Filter out None values when calculating averages
                    valid_humidities = [h for h in humidities[-10:] if h is not None] if humidities else []
                    avg_humidity = sum(valid_humidities) / len(valid_humidities) if valid_humidities else 50
                    avg_temp = sum(t for t in temperatures[-10:] if t) / len([t for t in temperatures[-10:] if t]) if temperatures else 20
                    
                    # Get previous temperature (last available temperature)
                    prev_temp = temperatures[-1] if temperatures else avg_temp

                    features = {
                        'humidity': avg_humidity,
                        'pressure': 1013,  # Default
                        'wind_speed': 5,   # Default
                        'hour_of_day': prediction_time.hour,
                        'month': prediction_time.month,
                        'prev_temp': prev_temp  # Add previous temperature for 6-feature model
                    }

                    temp_prediction = AIModelController.predict_temperature(features)

                    # Get or create model instance
                    temp_model = MLModel.objects.filter(model_type='TEMPERATURE_FORECAST', is_active=True).first()
                    if not temp_model:
                        # Create default model if it doesn't exist
                        temp_model = MLModel.objects.create(
                            name='Temperature Forecast Model',
                            model_type='TEMPERATURE_FORECAST',
                            status='READY',
                            is_active=True,
                            created_by=request.user
                        )

                    prediction = MLPrediction.objects.create(
                        model=temp_model,
                        prediction_type='TEMPERATURE_FORECAST',
                        prediction_value=temp_prediction['predicted_temperature'],
                        confidence_score=0.85,  # Placeholder
                        prediction_range=temp_prediction['confidence_range'],
                        target_timestamp=prediction_time,
                        input_data=features,
                        created_by=request.user
                    )

                    predictions.append(prediction)
            except Exception as temp_pred_error:
                # If temperature prediction fails (model not trained), skip temperature predictions
                messages.warning(request, f'Temperature prediction unavailable: {str(temp_pred_error)}. Please train the temperature model first. Continuing with anomaly detection only.')
                predictions = []

            # Create anomaly predictions
            anomaly_predictions = []
            processed_indices = 0  # Track which temperature values we've processed
            
            for data_point in recent_data[:min(20, len(temperatures))]:  # Match with available temperatures
                # Get temperature value (handle both formats)
                temp_value = None
                if data_point.temp_max is not None:
                    temp_value = float(data_point.temp_max)
                elif data_point.temperature is not None:
                    temp_value = float(data_point.temperature)
                
                if temp_value is not None and processed_indices < len(anomaly_results.get('anomaly_scores', [])):
                    anomaly_score = anomaly_results['anomaly_scores'][processed_indices]
                    is_anomaly = anomaly_results['is_anomaly'][processed_indices] if processed_indices < len(anomaly_results.get('is_anomaly', [])) else False

                    input_data = {'temperature': temp_value}
                    if data_point.humidity is not None:
                        input_data['humidity'] = float(data_point.humidity)

                    # Get or create anomaly detection model
                    anomaly_model = MLModel.objects.filter(model_type='ANOMALY_DETECTION', is_active=True).first()
                    if not anomaly_model:
                        # Create default model if it doesn't exist
                        anomaly_model = MLModel.objects.create(
                            name='Anomaly Detection Model',
                            model_type='ANOMALY_DETECTION',
                            status='READY',
                            is_active=True,
                            created_by=request.user
                        )

                    anomaly_pred = MLPrediction.objects.create(
                        model=anomaly_model,
                        prediction_type='ANOMALY_SCORE',
                        prediction_value=anomaly_score,
                        confidence_score=0.90,
                        target_timestamp=data_point.timestamp,
                        input_data=input_data,
                        climate_data=data_point,
                        is_anomaly=is_anomaly,
                        created_by=request.user
                    )

                    anomaly_predictions.append(anomaly_pred)
                    processed_indices += 1

            # Add confidence percentage to predictions for template display
            latest_predictions = predictions[:3]
            for prediction in latest_predictions:
                if prediction.confidence_score:
                    prediction.confidence_percentage = int(prediction.confidence_score * 100)
                else:
                    prediction.confidence_percentage = 0

            # Store analysis results
            context = {
                'analysis_completed': True,
                'total_data_points': len(recent_data),
                'predictions_generated': len(predictions),
                'anomalies_detected': sum(1 for p in anomaly_predictions if p.is_anomaly),
                'anomaly_threshold': anomaly_results.get('threshold', 0.1),
                'latest_predictions': latest_predictions,
                'recent_anomalies': [p for p in anomaly_predictions if p.is_anomaly][:5],
            }

            messages.success(request, f'AI Analysis completed! Generated {len(predictions)} predictions and analyzed {len(anomaly_predictions)} data points.')
            return render(request, 'ai_models/analysis_results.html', context)

        except Exception as e:
            messages.error(request, f'AI Analysis failed: {str(e)}')
            return redirect('dashboard:dashboard')

    @staticmethod
    @login_required
    def predictions_history_view(request):
        """View prediction history"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        predictions = MLPrediction.objects.select_related('model', 'created_by').order_by('-created_at')

        # Filter by type if specified
        prediction_type = request.GET.get('type')
        if prediction_type:
            predictions = predictions.filter(prediction_type=prediction_type)

        # Filter by model if specified
        model_id = request.GET.get('model')
        if model_id:
            predictions = predictions.filter(model_id=model_id)

        # Pagination (simplified)
        predictions = predictions[:100]  # Limit to recent 100

        # Add confidence percentage to predictions
        for prediction in predictions:
            if prediction.confidence_score:
                prediction.confidence_percentage = int(prediction.confidence_score * 100)
            else:
                prediction.confidence_percentage = 0

        context = {
            'predictions': predictions,
            'prediction_types': MLPrediction.PredictionType.choices,
            'models': MLModel.objects.filter(is_active=True),
            'current_type': prediction_type,
            'current_model': model_id
        }

        return render(request, 'ai_models/predictions_history.html', context)

    @staticmethod
    @login_required
    def model_performance_view(request):
        """View ML model performance metrics"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')

        # Get models with their metrics - show model info even without separate metrics records
        model_metrics = []
        for model in MLModel.objects.filter(is_active=True):
            # Try to get separate metrics record first
            latest_metrics = MLModelMetrics.objects.filter(model=model).order_by('-created_at').first()
            
            if latest_metrics:
                # Use separate metrics record
                model_metrics.append({
                    'model': model,
                    'metrics': latest_metrics,
                    'performance_score': AIModelController._calculate_performance_score(latest_metrics),
                    'has_separate_metrics': True
                })
            else:
                # Create metrics from model's own fields if available
                # This handles the case where models are trained but metrics weren't stored separately
                from django.utils import timezone
                from datetime import timedelta
                
                # Create a metrics-like object from model's stored scores
                class ModelMetricsProxy:
                    def __init__(self, model):
                        self.model = model
                        self.accuracy = model.accuracy_score
                        self.f1_score = model.f1_score
                        self.precision = None  # Not stored on model
                        self.recall = None  # Not stored on model
                        self.mse = model.mse_score
                        self.rmse = None
                        self.mae = None
                        self.created_at = model.last_trained or model.updated_at or timezone.now()
                        self.evaluation_start = model.training_data_start or (timezone.now() - timedelta(days=30))
                        self.evaluation_end = model.training_data_end or timezone.now()
                        self.total_predictions = MLPrediction.objects.filter(model=model).count()
                        self.correct_predictions = None
                        self.false_positives = None
                        self.false_negatives = None
                
                proxy_metrics = ModelMetricsProxy(model)
                
                # Calculate performance score from available data
                performance_score = None
                if model.accuracy_score is not None:
                    performance_score = model.accuracy_score
                elif model.f1_score is not None:
                    performance_score = model.f1_score
                elif model.mse_score is not None:
                    # Convert MSE to a score (lower is better, so invert it)
                    # Assuming MSE is reasonable, convert to 0-1 scale
                    performance_score = max(0, min(1, 1 - (model.mse_score / 100)))
                
                # If no metrics stored, calculate basic stats from predictions
                predictions = MLPrediction.objects.filter(model=model)
                if predictions.exists() and performance_score is None:
                    # Calculate average confidence as a proxy for performance
                    confidences = [p.confidence_score for p in predictions if p.confidence_score is not None]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        performance_score = avg_confidence
                        proxy_metrics.accuracy = avg_confidence  # Use confidence as accuracy proxy
                
                model_metrics.append({
                    'model': model,
                    'metrics': proxy_metrics,
                    'performance_score': performance_score,
                    'has_separate_metrics': False
                })

        context = {
            'model_metrics': model_metrics
        }

        return render(request, 'ai_models/model_performance.html', context)

    @staticmethod
    def _calculate_performance_score(metrics):
        """Calculate overall performance score for a model"""
        scores = []

        if metrics.accuracy is not None:
            scores.append(metrics.accuracy)
        if metrics.f1_score is not None:
            scores.append(metrics.f1_score)
        if metrics.precision is not None and metrics.recall is not None:
            harmonic_mean = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            scores.append(harmonic_mean)

        return sum(scores) / len(scores) if scores else None

    @staticmethod
    def initialize_ml_models():
        """Initialize ML models in database from trained files"""
        models_config = [
            {
                'name': 'Climate Anomaly Detection Model',
                'model_type': 'ANOMALY_DETECTION',
                'model_path': 'MLModels/anomaly_model.pkl',
                'description': 'Isolation Forest model for detecting climate anomalies'
            },
            {
                'name': 'Temperature Prediction Model',
                'model_type': 'TEMPERATURE_FORECAST',
                'model_path': 'MLModels/temperature_model.pkl',
                'description': 'Regression model for temperature forecasting'
            }
        ]

        created_models = []
        for config in models_config:
            model, created = MLModel.objects.get_or_create(
                model_type=config['model_type'],
                defaults={
                    'name': config['name'],
                    'model_path': config['model_path'],
                    'status': 'READY',
                    'is_active': True
                }
            )
            if created:
                created_models.append(model.name)

        return created_models

    @staticmethod
    def train_models(user=None, use_all_data=False):
        """Train both ML models using available climate data and save them with current NumPy version"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import os
        
        # Ensure MLModels directory exists
        models_dir = os.path.join(settings.BASE_DIR, 'MLModels')
        os.makedirs(models_dir, exist_ok=True)
        
        results = {
            'anomaly_model': {'success': False, 'error': None, 'metrics': {}},
            'temperature_model': {'success': False, 'error': None, 'metrics': {}}
        }
        
        try:
            # Get climate data for training
            if use_all_data:
                climate_data = ClimateData.objects.all().order_by('timestamp')
            else:
                # Use recent data (last 90 days) or all if less than 90 days worth
                recent_data = ClimateData.objects.filter(
                    timestamp__gte=timezone.now() - timedelta(days=90)
                ).order_by('timestamp')
                if recent_data.count() < 100:
                    climate_data = ClimateData.objects.all().order_by('timestamp')
                else:
                    climate_data = recent_data
            
            if not climate_data.exists():
                raise ValueError("No climate data available for training. Please upload data first.")
            
            # Prepare training data
            temperatures = []
            humidities = []
            feature_data = []  # For temperature prediction
            target_temps = []   # For temperature prediction
            
            for data in climate_data:
                # Get temperature (handle both formats)
                temp_value = None
                if data.temp_max is not None:
                    temp_value = float(data.temp_max)
                elif data.temperature is not None:
                    temp_value = float(data.temperature)
                
                if temp_value is not None:
                    temperatures.append(temp_value)
                    
                    # For temperature prediction: use previous temp to predict current
                    if len(temperatures) > 1:
                        prev_temp = temperatures[-2]
                        humidity = float(data.humidity) if data.humidity is not None else 50.0
                        pressure = float(data.atmospheric_pressure) if data.atmospheric_pressure is not None else 1013.0
                        wind = float(data.wind_speed) if data.wind_speed is not None else 5.0
                        hour = data.timestamp.hour
                        month = data.timestamp.month
                        
                        feature_data.append([humidity, pressure, wind, hour, month, prev_temp])
                        target_temps.append(temp_value)
                
                if data.humidity is not None:
                    humidities.append(float(data.humidity))
            
            # Need minimum data points
            if len(temperatures) < 10:
                raise ValueError(f"Insufficient data for training. Need at least 10 data points, got {len(temperatures)}")
            
            # ===== TRAIN ANOMALY DETECTION MODEL =====
            try:
                # Prepare features for anomaly detection
                if humidities and len(humidities) == len(temperatures):
                    # Use both temperature and humidity
                    features = np.column_stack([temperatures, humidities])
                else:
                    # Use only temperature
                    features = np.array(temperatures).reshape(-1, 1)
                
                # Scale features
                scaler_anomaly = StandardScaler()
                features_scaled = scaler_anomaly.fit_transform(features)
                
                # Train Isolation Forest model
                anomaly_model = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                anomaly_model.fit(features_scaled)
                
                # Evaluate (simple evaluation using training data)
                predictions = anomaly_model.predict(features_scaled)
                anomaly_score = (predictions == -1).sum() / len(predictions)
                
                # Save model with current NumPy version
                anomaly_path = os.path.join(models_dir, 'anomaly_model.pkl')
                with open(anomaly_path, 'wb') as f:
                    pickle.dump({
                        'model': anomaly_model,
                        'scaler': scaler_anomaly,
                        'trained_at': timezone.now().isoformat(),
                        'numpy_version': np.__version__,
                        'python_version': __import__('sys').version
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update database record
                anomaly_db_model = MLModel.objects.filter(model_type='ANOMALY_DETECTION').first()
                if not anomaly_db_model:
                    anomaly_db_model = MLModel.objects.create(
                        name='Climate Anomaly Detection Model',
                        model_type='ANOMALY_DETECTION',
                        model_path=anomaly_path,
                        status='READY',
                        is_active=True,
                        created_by=user
                    )
                else:
                    anomaly_db_model.model_path = anomaly_path
                    anomaly_db_model.status = 'READY'
                    anomaly_db_model.last_trained = timezone.now()
                    anomaly_db_model.training_data_start = climate_data.first().timestamp if climate_data.exists() else None
                    anomaly_db_model.training_data_end = climate_data.last().timestamp if climate_data.exists() else None
                    anomaly_db_model.save()
                
                # Clear cached model
                if 'anomaly' in AIModelController._loaded_models:
                    del AIModelController._loaded_models['anomaly']
                
                results['anomaly_model'] = {
                    'success': True,
                    'error': None,
                    'metrics': {
                        'anomaly_rate': float(anomaly_score),
                        'training_samples': len(temperatures)
                    }
                }
                
            except Exception as e:
                results['anomaly_model'] = {
                    'success': False,
                    'error': str(e),
                    'metrics': {}
                }
            
            # ===== TRAIN TEMPERATURE PREDICTION MODEL =====
            try:
                if len(feature_data) < 10:
                    raise ValueError(f"Insufficient data for temperature prediction. Need at least 10 data points, got {len(feature_data)}")
                
                # Prepare features and targets
                X = np.array(feature_data)
                y = np.array(target_temps)
                
                # Scale features
                scaler_temp = StandardScaler()
                X_scaled = scaler_temp.fit_transform(X)
                
                # Train Linear Regression model
                temp_model = LinearRegression()
                temp_model.fit(X_scaled, y)
                
                # Evaluate model
                y_pred = temp_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mse)
                
                # Calculate R² score
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Save model with current NumPy version
                temp_path = os.path.join(models_dir, 'temperature_model.pkl')
                with open(temp_path, 'wb') as f:
                    pickle.dump({
                        'model': temp_model,
                        'scaler': scaler_temp,
                        'trained_at': timezone.now().isoformat(),
                        'numpy_version': np.__version__,
                        'python_version': __import__('sys').version,
                        'feature_names': ['humidity', 'pressure', 'wind_speed', 'hour_of_day', 'month', 'prev_temp']
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update database record
                temp_db_model = MLModel.objects.filter(model_type='TEMPERATURE_FORECAST').first()
                if not temp_db_model:
                    temp_db_model = MLModel.objects.create(
                        name='Temperature Prediction Model',
                        model_type='TEMPERATURE_FORECAST',
                        model_path=temp_path,
                        status='READY',
                        is_active=True,
                        mse_score=mse,
                        accuracy_score=r2_score,  # Using R² as accuracy proxy
                        last_trained=timezone.now(),
                        training_data_start=climate_data.first().timestamp if climate_data.exists() else None,
                        training_data_end=climate_data.last().timestamp if climate_data.exists() else None,
                        created_by=user
                    )
                else:
                    temp_db_model.model_path = temp_path
                    temp_db_model.status = 'READY'
                    temp_db_model.last_trained = timezone.now()
                    temp_db_model.mse_score = mse
                    temp_db_model.accuracy_score = r2_score
                    temp_db_model.training_data_start = climate_data.first().timestamp if climate_data.exists() else None
                    temp_db_model.training_data_end = climate_data.last().timestamp if climate_data.exists() else None
                    temp_db_model.save()
                
                # Clear cached model
                if 'temperature' in AIModelController._loaded_models:
                    del AIModelController._loaded_models['temperature']
                
                results['temperature_model'] = {
                    'success': True,
                    'error': None,
                    'metrics': {
                        'mse': float(mse),
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'r2_score': float(r2_score),
                        'training_samples': len(feature_data)
                    }
                }
                
            except Exception as e:
                results['temperature_model'] = {
                    'success': False,
                    'error': str(e),
                    'metrics': {}
                }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'anomaly_model': {'success': False, 'error': str(e)},
                'temperature_model': {'success': False, 'error': str(e)}
            }

    @staticmethod
    @login_required
    def train_models_view(request):
        """View to trigger model training"""
        if request.user.role not in ['ADMIN', 'ANALYST']:
            messages.error(request, 'Access denied. Requires Analyst or Admin privileges.')
            return redirect('dashboard:dashboard')
        
        use_all_data = request.POST.get('use_all_data', 'false') == 'true'
        
        try:
            results = AIModelController.train_models(user=request.user, use_all_data=use_all_data)
            
            if results.get('error'):
                messages.error(request, f'Training failed: {results["error"]}')
            else:
                success_count = sum(1 for m in ['anomaly_model', 'temperature_model'] 
                                  if results.get(m, {}).get('success', False))
                
                if success_count == 2:
                    messages.success(request, 
                        f'Models trained successfully! '
                        f'Anomaly Model: {results["anomaly_model"]["metrics"].get("training_samples", 0)} samples, '
                        f'Temperature Model: {results["temperature_model"]["metrics"].get("r2_score", 0):.3f} R² score')
                elif success_count == 1:
                    success_model = 'Anomaly' if results.get('anomaly_model', {}).get('success') else 'Temperature'
                    messages.warning(request, 
                        f'{success_model} model trained successfully, but the other model failed. '
                        f'Check details below.')
                else:
                    messages.error(request, 'Both models failed to train. Check the errors below.')
            
            # Pass results to template
            context = {
                'training_results': results,
                'use_all_data': use_all_data
            }
            
            return render(request, 'ai_models/training_results.html', context)
            
        except Exception as e:
            messages.error(request, f'Training error: {str(e)}')
            return redirect('ai_models:ai_dashboard')
