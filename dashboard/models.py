from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class DashboardWidget(models.Model):
    """Configurable dashboard widgets for data visualization"""

    class WidgetType(models.TextChoices):
        CHART = 'CHART', _('Chart')
        GAUGE = 'GAUGE', _('Gauge')
        TABLE = 'TABLE', _('Data Table')
        MAP = 'MAP', _('Map View')
        METRIC = 'METRIC', _('Key Metric')

    class DataSource(models.TextChoices):
        TEMPERATURE = 'TEMPERATURE', _('Temperature Data')
        HUMIDITY = 'HUMIDITY', _('Humidity Data')
        PRECIPITATION = 'PRECIPITATION', _('Precipitation Data')
        WIND = 'WIND', _('Wind Data')
        AIR_QUALITY = 'AIR_QUALITY', _('Air Quality Data')
        CUSTOM = 'CUSTOM', _('Custom Query')

    title = models.CharField(max_length=200, help_text="Widget title")
    widget_type = models.CharField(
        max_length=20,
        choices=WidgetType.choices,
        help_text="Type of visualization widget"
    )
    data_source = models.CharField(
        max_length=20,
        choices=DataSource.choices,
        help_text="Source of data for this widget"
    )

    # Position and size on dashboard
    position_x = models.PositiveIntegerField(default=0, help_text="X position on dashboard grid")
    position_y = models.PositiveIntegerField(default=0, help_text="Y position on dashboard grid")
    width = models.PositiveIntegerField(default=4, help_text="Widget width in grid units")
    height = models.PositiveIntegerField(default=3, help_text="Widget height in grid units")

    # Configuration
    config = models.JSONField(
        default=dict,
        help_text="Widget-specific configuration (chart type, filters, etc.)"
    )
    refresh_interval = models.PositiveIntegerField(
        default=300,
        help_text="Auto-refresh interval in seconds (0 = no auto-refresh)"
    )

    # Access control
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='dashboard_widgets'
    )
    is_public = models.BooleanField(
        default=False,
        help_text="Whether this widget is visible to all users"
    )

    is_active = models.BooleanField(default=True, help_text="Whether this widget is active")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Dashboard Widget'
        verbose_name_plural = 'Dashboard Widgets'
        ordering = ['position_y', 'position_x']

    def __str__(self):
        return f"{self.title} ({self.get_widget_type_display()})"


class UserDashboard(models.Model):
    """User-specific dashboard configurations"""

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='dashboard'
    )
    title = models.CharField(
        max_length=200,
        default="My Dashboard",
        help_text="Dashboard title"
    )
    layout_config = models.JSONField(
        default=dict,
        help_text="Dashboard layout configuration"
    )
    is_default = models.BooleanField(
        default=True,
        help_text="Whether this is the user's default dashboard"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Dashboard'
        verbose_name_plural = 'User Dashboards'

    def __str__(self):
        return f"{self.user.username}'s Dashboard"
