from django.contrib import admin
from .models import DashboardWidget, UserDashboard

@admin.register(DashboardWidget)
class DashboardWidgetAdmin(admin.ModelAdmin):
    """Admin interface for Dashboard Widgets"""

    list_display = ('title', 'widget_type', 'data_source', 'created_by', 'is_public', 'is_active', 'created_at')
    list_filter = ('widget_type', 'data_source', 'is_public', 'is_active', 'created_at')
    search_fields = ('title', 'created_by__username')

    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'widget_type', 'data_source', 'created_by')
        }),
        ('Layout & Positioning', {
            'fields': ('position_x', 'position_y', 'width', 'height'),
            'classes': ('collapse',)
        }),
        ('Configuration', {
            'fields': ('config', 'refresh_interval'),
            'classes': ('collapse',)
        }),
        ('Settings', {
            'fields': ('is_public', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(UserDashboard)
class UserDashboardAdmin(admin.ModelAdmin):
    """Admin interface for User Dashboards"""

    list_display = ('user', 'title', 'is_default', 'created_at')
    list_filter = ('is_default', 'created_at')
    search_fields = ('user__username', 'user__email', 'title')

    fieldsets = (
        ('Dashboard Owner', {
            'fields': ('user',)
        }),
        ('Dashboard Settings', {
            'fields': ('title', 'is_default')
        }),
        ('Layout Configuration', {
            'fields': ('layout_config',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at', 'updated_at')
