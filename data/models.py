from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator

User = get_user_model()

class DataSource(models.Model):
    """Represents different sources of climate data"""

    class SourceType(models.TextChoices):
        SATELLITE = 'SATELLITE', _('Satellite Imagery')
        WEATHER_STATION = 'WEATHER_STATION', _('Weather Station')
        SENSOR = 'SENSOR', _('Environmental Sensor')
        HISTORICAL = 'HISTORICAL', _('Historical Data')
        REALTIME = 'REALTIME', _('Real-time Data')

    name = models.CharField(max_length=200, help_text="Name of the data source")
    source_type = models.CharField(
        max_length=20,
        choices=SourceType.choices,
        help_text="Type of data source"
    )
    location = models.CharField(
        max_length=200,
        blank=True,
        help_text="Geographic location of the data source"
    )
    latitude = models.DecimalField(
        max_digits=10,
        decimal_places=7,
        null=True,
        blank=True,
        validators=[MinValueValidator(-90), MaxValueValidator(90)],
        help_text="Latitude coordinate"
    )
    longitude = models.DecimalField(
        max_digits=10,
        decimal_places=7,
        null=True,
        blank=True,
        validators=[MinValueValidator(-180), MaxValueValidator(180)],
        help_text="Longitude coordinate"
    )
    is_active = models.BooleanField(default=True, help_text="Whether this source is currently active")
    description = models.TextField(blank=True, help_text="Additional description")
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_data_sources'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Data Source'
        verbose_name_plural = 'Data Sources'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.get_source_type_display()})"


class ClimateData(models.Model):
    """Main model for storing climate data records"""

    data_source = models.ForeignKey(
        DataSource,
        on_delete=models.CASCADE,
        related_name='climate_data'
    )

    # Temporal information
    timestamp = models.DateTimeField(help_text="When the data was recorded")

    # Environmental measurements
    temperature = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Temperature in Celsius"
    )
    temp_max = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Maximum temperature in Celsius"
    )
    temp_min = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Minimum temperature in Celsius"
    )
    humidity = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Humidity percentage"
    )
    precipitation = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Precipitation in mm"
    )
    wind_speed = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Wind speed in m/s"
    )
    wind_direction = models.DecimalField(
        max_digits=5,
        decimal_places=1,
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(360)],
        help_text="Wind direction in degrees"
    )
    atmospheric_pressure = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Atmospheric pressure in hPa"
    )

    # Air quality indices
    co2_level = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="CO2 level in ppm"
    )
    pm25 = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="PM2.5 concentration"
    )
    pm10 = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="PM10 concentration"
    )

    # Weather description
    weather_description = models.CharField(
        max_length=100,
        blank=True,
        help_text="Weather condition description (e.g., rain, sunny, cloudy)"
    )

    # Additional metadata
    quality_score = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Data quality score (0-1)"
    )
    notes = models.TextField(blank=True, help_text="Additional notes about this data point")

    # Audit fields
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='uploaded_climate_data'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Climate Data'
        verbose_name_plural = 'Climate Data'
        ordering = ['-timestamp']
        get_latest_by = 'timestamp'
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['data_source', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.data_source.name} - {self.timestamp}"


class DataUpload(models.Model):
    """Tracks data upload batches for bulk data ingestion"""

    class Status(models.TextChoices):
        PENDING = 'PENDING', _('Pending')
        PROCESSING = 'PROCESSING', _('Processing')
        COMPLETED = 'COMPLETED', _('Completed')
        FAILED = 'FAILED', _('Failed')

    filename = models.CharField(max_length=255, help_text="Original filename")
    file_path = models.FileField(
        upload_to='data_uploads/%Y/%m/%d/',
        help_text="Path to uploaded file"
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    records_processed = models.PositiveIntegerField(default=0)
    errors = models.TextField(blank=True, help_text="Processing errors")
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='data_uploads'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Data Upload'
        verbose_name_plural = 'Data Uploads'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.filename} - {self.status}"
