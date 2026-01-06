# Fixed import
from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class AlertThreshold(models.Model):
    """Defines thresholds for climate data alerts"""

    class Metric(models.TextChoices):
        TEMPERATURE = 'TEMPERATURE', _('Temperature (°C)')
        HUMIDITY = 'HUMIDITY', _('Humidity (%)')
        PRECIPITATION = 'PRECIPITATION', _('Precipitation (mm)')
        WIND_SPEED = 'WIND_SPEED', _('Wind Speed (m/s)')
        CO2_LEVEL = 'CO2_LEVEL', _('CO2 Level (ppm)')
        PM25 = 'PM25', _('PM2.5')
        PM10 = 'PM10', _('PM10')

    class Condition(models.TextChoices):
        ABOVE = 'ABOVE', _('Above threshold')
        BELOW = 'BELOW', _('Below threshold')
        EQUALS = 'EQUALS', _('Equals threshold')

    name = models.CharField(max_length=200, help_text="Name of the alert threshold")
    metric = models.CharField(
        max_length=20,
        choices=Metric.choices,
        help_text="Climate metric to monitor"
    )
    condition = models.CharField(
        max_length=10,
        choices=Condition.choices,
        help_text="Condition for triggering alert"
    )
    threshold_value = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="Threshold value for the metric"
    )
    is_active = models.BooleanField(default=True, help_text="Whether this threshold is active")
    description = models.TextField(blank=True, help_text="Description of this threshold")
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_alert_thresholds'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Alert Threshold'
        verbose_name_plural = 'Alert Thresholds'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name}: {self.get_metric_display()} {self.get_condition_display()} {self.threshold_value}"


class Notification(models.Model):
    """System notifications and alerts"""

    class Type(models.TextChoices):
        ALERT = 'ALERT', _('Climate Alert')
        SYSTEM = 'SYSTEM', _('System Notification')
        DATA = 'DATA', _('Data Processing')
        SECURITY = 'SECURITY', _('Security Alert')

    class Priority(models.TextChoices):
        LOW = 'LOW', _('Low')
        MEDIUM = 'MEDIUM', _('Medium')
        HIGH = 'HIGH', _('High')
        CRITICAL = 'CRITICAL', _('Critical')

    title = models.CharField(max_length=200, help_text="Notification title")
    message = models.TextField(help_text="Notification message content")
    notification_type = models.CharField(
        max_length=20,
        choices=Type.choices,
        default=Type.SYSTEM
    )
    priority = models.CharField(
        max_length=20,
        choices=Priority.choices,
        default=Priority.MEDIUM
    )

    # Related data
    alert_threshold = models.ForeignKey(
        AlertThreshold,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='notifications'
    )
    climate_data = models.ForeignKey(
        'data.ClimateData',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='notifications'
    )

    # Recipients
    recipients = models.ManyToManyField(
        User,
        related_name='notifications',
        help_text="Users who should receive this notification"
    )

    # Status
    is_read = models.BooleanField(default=False, help_text="Whether the notification has been read")
    sent_at = models.DateTimeField(null=True, blank=True, help_text="When the notification was sent")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Notification'
        verbose_name_plural = 'Notifications'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.notification_type}: {self.title}"


class Feedback(models.Model):
    """User feedback and support requests"""

    class Category(models.TextChoices):
        BUG = 'BUG', _('Bug Report')
        FEATURE = 'FEATURE', _('Feature Request')
        DATA = 'DATA', _('Data Issue')
        USABILITY = 'USABILITY', _('Usability Issue')
        OTHER = 'OTHER', _('Other')

    class Status(models.TextChoices):
        OPEN = 'OPEN', _('Open')
        IN_PROGRESS = 'IN_PROGRESS', _('In Progress')
        RESOLVED = 'RESOLVED', _('Resolved')
        CLOSED = 'CLOSED', _('Closed')

    subject = models.CharField(max_length=200, help_text="Feedback subject")
    category = models.CharField(
        max_length=20,
        choices=Category.choices,
        default=Category.OTHER
    )
    description = models.TextField(help_text="Detailed description of the feedback")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OPEN
    )

    submitted_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='feedback_submissions'
    )
    assigned_to = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_feedback'
    )

    # Resolution details
    resolution = models.TextField(blank=True, help_text="Resolution details")
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_feedback'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Feedback'
        verbose_name_plural = 'Feedback'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.category}: {self.subject}"
