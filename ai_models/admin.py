from django.contrib import admin
from .models import MLModel, MLPrediction, MLModelMetrics

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    """Admin interface for ML Models"""

    list_display = ('name', 'model_type', 'status', 'accuracy_score', 'is_active', 'last_trained', 'created_by')
    list_filter = ('model_type', 'status', 'is_active', 'created_at')
    search_fields = ('name', 'model_type', 'created_by__username')

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'model_type', 'status')
        }),
        ('Model Details', {
            'fields': ('model_file', 'model_path', 'config')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy_score', 'f1_score', 'mse_score')
        }),
        ('Training Information', {
            'fields': ('training_data_start', 'training_data_end', 'last_trained')
        }),
        ('Management', {
            'fields': ('created_by', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at', 'updated_at')

@admin.register(MLPrediction)
class MLPredictionAdmin(admin.ModelAdmin):
    """Admin interface for ML Predictions"""

    list_display = ('model', 'prediction_type', 'prediction_value', 'is_anomaly', 'target_timestamp', 'created_by')
    list_filter = ('prediction_type', 'is_anomaly', 'created_at', 'model')
    search_fields = ('model__name', 'created_by__username')

    fieldsets = (
        ('Prediction Details', {
            'fields': ('model', 'prediction_type', 'prediction_value', 'confidence_score')
        }),
        ('Prediction Range', {
            'fields': ('prediction_range', 'error_margin')
        }),
        ('Context', {
            'fields': ('target_timestamp', 'input_data', 'climate_data')
        }),
        ('Results', {
            'fields': ('actual_value', 'is_anomaly', 'notes')
        }),
        ('Management', {
            'fields': ('created_by',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at',)

@admin.register(MLModelMetrics)
class MLModelMetricsAdmin(admin.ModelAdmin):
    """Admin interface for ML Model Metrics"""

    list_display = ('model', 'evaluation_start', 'evaluation_end', 'accuracy', 'f1_score', 'evaluated_by')
    list_filter = ('model', 'evaluation_start', 'evaluated_by')
    search_fields = ('model__name', 'evaluated_by__username')

    fieldsets = (
        ('Model & Period', {
            'fields': ('model', 'evaluation_start', 'evaluation_end')
        }),
        ('Classification Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Regression Metrics', {
            'fields': ('mse', 'rmse', 'mae')
        }),
        ('Prediction Results', {
            'fields': ('total_predictions', 'correct_predictions', 'false_positives', 'false_negatives')
        }),
        ('Model Health', {
            'fields': ('model_drift_score', 'evaluated_by')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at',)