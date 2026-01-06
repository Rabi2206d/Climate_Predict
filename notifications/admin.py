from django.contrib import admin
from .models import AlertThreshold, Notification, Feedback

@admin.register(AlertThreshold)
class AlertThresholdAdmin(admin.ModelAdmin):
    """Admin interface for Alert Thresholds"""

    list_display = ('name', 'metric', 'condition', 'threshold_value', 'is_active', 'created_by', 'created_at')
    list_filter = ('metric', 'condition', 'is_active', 'created_at')
    search_fields = ('name', 'description', 'created_by__username')

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'is_active')
        }),
        ('Alert Configuration', {
            'fields': ('metric', 'condition', 'threshold_value')
        }),
        ('Management', {
            'fields': ('created_by',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    """Admin interface for Notifications"""

    list_display = ('title', 'notification_type', 'priority', 'is_read', 'created_at', 'sent_at')
    list_filter = ('notification_type', 'priority', 'is_read', 'created_at')
    search_fields = ('title', 'message')
    readonly_fields = ('created_at', 'sent_at')

    fieldsets = (
        ('Notification Details', {
            'fields': ('title', 'message', 'notification_type', 'priority')
        }),
        ('Related Data', {
            'fields': ('alert_threshold', 'climate_data'),
            'classes': ('collapse',)
        }),
        ('Recipients', {
            'fields': ('recipients',)
        }),
        ('Status', {
            'fields': ('is_read', 'sent_at', 'created_at'),
            'classes': ('collapse',)
        }),
    )

    filter_horizontal = ('recipients',)

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    """Admin interface for User Feedback"""

    list_display = ('subject', 'category', 'status', 'submitted_by', 'assigned_to', 'created_at')
    list_filter = ('category', 'status', 'created_at')
    search_fields = ('subject', 'description', 'submitted_by__username', 'assigned_to__username')

    fieldsets = (
        ('Feedback Details', {
            'fields': ('subject', 'category', 'description')
        }),
        ('Status & Assignment', {
            'fields': ('status', 'submitted_by', 'assigned_to')
        }),
        ('Resolution', {
            'fields': ('resolution', 'resolved_by', 'resolved_at'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at', 'updated_at')
