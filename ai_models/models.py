from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class MLModel(models.Model):
    """Stores information about trained ML models"""

    class ModelType(models.TextChoices):
        ANOMALY_DETECTION = 'ANOMALY', _('Anomaly Detection')
        TEMPERATURE_FORECAST = 'TEMPERATURE', _('Temperature Forecast')
        CLIMATE_PREDICTION = 'CLIMATE', _('Climate Prediction')
        AIR_QUALITY_FORECAST = 'AIR_QUALITY', _('Air Quality Forecast')

    class ModelStatus(models.TextChoices):
        TRAINING = 'TRAINING', _('Training')
        READY = 'READY', _('Ready')
        FAILED = 'FAILED', _('Failed')
        OUTDATED = 'OUTDATED', _('Outdated')

    name = models.CharField(max_length=200, help_text="Descriptive name for the model")
    model_type = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        help_text="Type of ML model"
    )
    status = models.CharField(
        max_length=20,
        choices=ModelStatus.choices,
        default=ModelStatus.READY,
        help_text="Current status of the model"
    )

    # Model file information
    model_file = models.FileField(
        upload_to='ml_models/',
        help_text="Path to the trained model file"
    )
    model_path = models.CharField(
        max_length=500,
        help_text="Full path to model file for loading"
    )

    # Model metadata
    accuracy_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Model accuracy score (0-1)"
    )
    f1_score = models.FloatField(
        null=True,
        blank=True,
        help_text="F1 score for classification models"
    )
    mse_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Mean Squared Error for regression models"
    )

    # Training information
    training_data_start = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Start date of training data"
    )
    training_data_end = models.DateTimeField(
        null=True,
        blank=True,
        help_text="End date of training data"
    )
    last_trained = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the model was last trained"
    )

    # Model configuration
    config = models.JSONField(
        default=dict,
        help_text="Model configuration parameters (JSON)"
    )

    # Management
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_ml_models'
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this model is available for predictions"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'ML Model'
        verbose_name_plural = 'ML Models'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"


class MLPrediction(models.Model):
    """Stores ML model predictions and results"""

    class PredictionType(models.TextChoices):
        ANOMALY_SCORE = 'ANOMALY_SCORE', _('Anomaly Score')
        TEMPERATURE_FORECAST = 'TEMPERATURE_FORECAST', _('Temperature Forecast')
        CLIMATE_TREND = 'CLIMATE_TREND', _('Climate Trend')
        AIR_QUALITY_FORECAST = 'AIR_QUALITY_FORECAST', _('Air Quality Forecast')

    model = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name='predictions'
    )

    prediction_type = models.CharField(
        max_length=30,
        choices=PredictionType.choices,
        help_text="Type of prediction made"
    )

    # Prediction data
    prediction_value = models.FloatField(
        help_text="The predicted value"
    )
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence score for the prediction (0-1)"
    )
    prediction_range = models.JSONField(
        default=dict,
        help_text="Prediction range/confidence intervals (JSON)"
    )

    # Context data (what the model was predicting about)
    target_timestamp = models.DateTimeField(
        help_text="Timestamp the prediction is for"
    )
    input_data = models.JSONField(
        default=dict,
        help_text="Input data used for prediction (JSON)"
    )

    # Related climate data
    climate_data = models.ForeignKey(
        'data.ClimateData',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='ml_predictions'
    )

    # Results and validation
    actual_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Actual observed value (for validation)"
    )
    error_margin = models.FloatField(
        null=True,
        blank=True,
        help_text="Prediction error margin"
    )
    is_anomaly = models.BooleanField(
        default=False,
        help_text="Whether this prediction indicates an anomaly"
    )

    # Metadata
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about this prediction"
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='ml_predictions'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'ML Prediction'
        verbose_name_plural = 'ML Predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model', 'target_timestamp']),
            models.Index(fields=['prediction_type', 'created_at']),
        ]

    def __str__(self):
        return f"{self.model.name}: {self.prediction_value} ({self.target_timestamp})"


class MLModelMetrics(models.Model):
    """Stores performance metrics for ML models over time"""

    model = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name='metrics'
    )

    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)

    # Evaluation period
    evaluation_start = models.DateTimeField()
    evaluation_end = models.DateTimeField()

    # Additional metrics
    total_predictions = models.PositiveIntegerField(default=0)
    correct_predictions = models.PositiveIntegerField(default=0)
    false_positives = models.PositiveIntegerField(default=0)
    false_negatives = models.PositiveIntegerField(default=0)

    # Model health
    model_drift_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Measure of how much the model has drifted from training data"
    )

    evaluated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='model_evaluations'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'ML Model Metrics'
        verbose_name_plural = 'ML Model Metrics'
        ordering = ['-created_at']
        unique_together = ['model', 'evaluation_start', 'evaluation_end']

    def __str__(self):
        return f"{self.model.name} metrics ({self.evaluation_start.date()} - {self.evaluation_end.date()})"
