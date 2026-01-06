from django.contrib import admin
from .models import DataSource, ClimateData, DataUpload

@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    """Admin interface for Data Sources"""

    list_display = ('name', 'source_type', 'location', 'latitude', 'longitude', 'is_active', 'created_by', 'created_at')
    list_filter = ('source_type', 'is_active', 'created_at')
    search_fields = ('name', 'location', 'description')
    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'source_type', 'description')
        }),
        ('Location', {
            'fields': ('location', 'latitude', 'longitude')
        }),
        ('Settings', {
            'fields': ('is_active', 'created_by')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ClimateData)
class ClimateDataAdmin(admin.ModelAdmin):
    """Admin interface for Climate Data"""

    list_display = ('data_source', 'timestamp', 'temperature', 'humidity', 'precipitation', 'quality_score', 'uploaded_by')
    list_filter = ('data_source', 'timestamp', 'quality_score', 'uploaded_by')
    search_fields = ('data_source__name', 'notes')
    readonly_fields = ('uploaded_at',)

    fieldsets = (
        ('Data Source', {
            'fields': ('data_source', 'uploaded_by')
        }),
        ('Temporal Information', {
            'fields': ('timestamp',)
        }),
        ('Environmental Measurements', {
            'fields': ('temperature', 'humidity', 'precipitation', 'wind_speed', 'wind_direction', 'atmospheric_pressure'),
            'classes': ('collapse',)
        }),
        ('Air Quality', {
            'fields': ('co2_level', 'pm25', 'pm10'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('quality_score', 'notes', 'uploaded_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    """Admin interface for Data Uploads"""

    list_display = ('filename', 'status', 'records_processed', 'uploaded_by', 'created_at', 'completed_at')
    list_filter = ('status', 'created_at', 'uploaded_by')
    search_fields = ('filename', 'uploaded_by__username')
    readonly_fields = ('created_at', 'completed_at')

    fieldsets = (
        ('File Information', {
            'fields': ('filename', 'file_path', 'uploaded_by')
        }),
        ('Processing Status', {
            'fields': ('status', 'records_processed', 'errors')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
