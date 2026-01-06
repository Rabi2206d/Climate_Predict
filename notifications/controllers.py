"""
Notifications Controllers - Handle business logic for alerts and notifications
"""
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Count
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from .models import AlertThreshold, Notification, Feedback

class NotificationsController:
    """Controller for notifications and alerts operations"""

    @staticmethod
    def get_user_notifications(user, unread_only=False):
        """Get notifications for a specific user"""
        queryset = Notification.objects.filter(recipients=user)

        if unread_only:
            queryset = queryset.filter(is_read=False)

        return queryset.order_by('-created_at')

    @staticmethod
    def check_alert_thresholds():
        """Check all active alert thresholds and create notifications if triggered"""
        from data.models import ClimateData

        now = timezone.now()
        last_check = now - timedelta(minutes=15)  # Check last 15 minutes of data

        new_notifications = []

        for threshold in AlertThreshold.objects.filter(is_active=True):
            # Get recent data that might trigger this alert
            recent_data = ClimateData.objects.filter(
                timestamp__gte=last_check
            )

            # Apply metric-specific filtering - handle both temperature formats
            field_names = []
            if threshold.metric == 'TEMPERATURE':
                field_names = ['temperature', 'temp_max']  # Try both formats
            elif threshold.metric == 'HUMIDITY':
                field_names = ['humidity']
            elif threshold.metric == 'PRECIPITATION':
                field_names = ['precipitation']
            elif threshold.metric == 'WIND_SPEED':
                field_names = ['wind_speed']
            elif threshold.metric == 'CO2_LEVEL':
                field_names = ['co2_level']
            elif threshold.metric == 'PM25':
                field_names = ['pm25']
            elif threshold.metric == 'PM10':
                field_names = ['pm10']
            else:
                continue

            # Check each possible field name
            triggered_data = None
            for field_name in field_names:
                # Build filter based on condition
                filter_kwargs = {f'{field_name}__isnull': False}
                if threshold.condition == 'ABOVE':
                    filter_kwargs[f'{field_name}__gt'] = threshold.threshold_value
                elif threshold.condition == 'BELOW':
                    filter_kwargs[f'{field_name}__lt'] = threshold.threshold_value
                elif threshold.condition == 'EQUALS':
                    filter_kwargs[f'{field_name}'] = threshold.threshold_value

                field_data = recent_data.filter(**filter_kwargs)
                if field_data.exists():
                    if triggered_data is None:
                        triggered_data = field_data
                    else:
                        triggered_data = triggered_data | field_data
                    break  # Found data, no need to check other fields

            if triggered_data is None:
                continue

            if triggered_data.exists():
                # Create notification for each triggered reading
                for data_point in triggered_data:
                    # Get the actual value that triggered the alert
                    trigger_value = None
                    for field_name in field_names:
                        value = getattr(data_point, field_name, None)
                        if value is not None:
                            trigger_value = value
                            break
                    
                    if trigger_value is None:
                        continue
                    
                    notification = Notification.objects.create(
                        title=f"Alert: {threshold.name}",
                        message=f"{threshold.get_metric_display()} {threshold.get_condition_display()} {threshold.threshold_value} "
                               f"at {data_point.data_source.name} "
                               f"(Value: {trigger_value})",
                        notification_type='ALERT',
                        priority='HIGH',
                        alert_threshold=threshold,
                        climate_data=data_point,
                    )

                    # Add recipients (all users with appropriate roles)
                    from django.contrib.auth import get_user_model
                    User = get_user_model()
                    recipients = User.objects.filter(
                        Q(role='ADMIN') | Q(role='ANALYST')
                    )
                    notification.recipients.set(recipients)
                    notification.save()

                    new_notifications.append(notification)

        return new_notifications

    @staticmethod
    def create_system_notification(title, message, recipients=None, priority='MEDIUM'):
        """Create a system notification"""
        notification = Notification.objects.create(
            title=title,
            message=message,
            notification_type='SYSTEM',
            priority=priority,
        )

        if recipients:
            notification.recipients.set(recipients)
        else:
            # Send to all active users
            from django.contrib.auth import get_user_model
            User = get_user_model()
            notification.recipients.set(User.objects.filter(is_active=True))

        notification.save()
        return notification

    @staticmethod
    def get_alert_thresholds():
        """Get all alert thresholds with statistics"""
        return AlertThreshold.objects.annotate(
            active_alerts=Count(
                'notifications',
                filter=Q(notifications__created_at__gte=timezone.now() - timedelta(days=7))
            )
        ).order_by('-is_active', '-created_at')

    @staticmethod
    def create_alert_threshold(data, created_by):
        """Create a new alert threshold"""
        threshold = AlertThreshold.objects.create(
            name=data['name'],
            metric=data['metric'],
            condition=data['condition'],
            threshold_value=data['threshold_value'],
            is_active=data.get('is_active', True),
            description=data.get('description', ''),
            created_by=created_by
        )
        return threshold

    @staticmethod
    def update_alert_threshold(threshold_id, data, updated_by):
        """Update an existing alert threshold"""
        threshold = AlertThreshold.objects.get(id=threshold_id)
        for field, value in data.items():
            if hasattr(threshold, field):
                setattr(threshold, field, value)
        threshold.save()
        return threshold

    @staticmethod
    def get_feedback_statistics():
        """Get feedback statistics"""
        total_feedback = Feedback.objects.count()
        open_feedback = Feedback.objects.filter(status='OPEN').count()
        resolved_feedback = Feedback.objects.filter(status='RESOLVED').count()

        category_stats = Feedback.objects.values('category').annotate(
            count=Count('id')
        ).order_by('-count')

        return {
            'total_feedback': total_feedback,
            'open_feedback': open_feedback,
            'resolved_feedback': resolved_feedback,
            'resolution_rate': (resolved_feedback / max(total_feedback, 1)) * 100,
            'category_stats': list(category_stats),
        }

    @staticmethod
    def submit_feedback(data, submitted_by):
        """Submit user feedback"""
        feedback = Feedback.objects.create(
            subject=data['subject'],
            category=data['category'],
            description=data['description'],
            submitted_by=submitted_by
        )
        return feedback

    @staticmethod
    def resolve_feedback(feedback_id, resolution, resolved_by):
        """Resolve a feedback item"""
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.status = 'RESOLVED'
        feedback.resolution = resolution
        feedback.resolved_at = timezone.now()
        feedback.resolved_by = resolved_by
        feedback.save()
        return feedback

    @staticmethod
    def get_notification_summary(user):
        """Get notification summary for a user"""
        unread_count = Notification.objects.filter(
            recipients=user,
            is_read=False
        ).count()

        recent_notifications = Notification.objects.filter(
            recipients=user
        ).order_by('-created_at')[:5]

        alerts_this_week = Notification.objects.filter(
            recipients=user,
            notification_type='ALERT',
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count()

        return {
            'unread_count': unread_count,
            'recent_notifications': recent_notifications,
            'alerts_this_week': alerts_this_week,
        }

    @staticmethod
    def mark_notifications_read(user, notification_ids=None):
        """Mark notifications as read"""
        queryset = Notification.objects.filter(recipients=user, is_read=False)

        if notification_ids:
            queryset = queryset.filter(id__in=notification_ids)

        updated_count = queryset.update(is_read=True, sent_at=timezone.now())
        return updated_count

    @staticmethod
    def cleanup_old_notifications(days=30):
        """Clean up old notifications (soft delete or archive)"""
        cutoff_date = timezone.now() - timedelta(days=days)

        # Mark old notifications as read if they're not critical
        old_notifications = Notification.objects.filter(
            created_at__lt=cutoff_date,
            is_read=False
        ).exclude(priority='CRITICAL')

        updated_count = old_notifications.update(is_read=True)
        return updated_count

    @staticmethod
    @login_required
    def notifications_view(request):
        """Main notifications dashboard"""
        # Get notification summary for the user
        summary = NotificationsController.get_notification_summary(request.user)

        # Get user's notifications
        notifications = NotificationsController.get_user_notifications(request.user)

        context = {
            'notifications': notifications,
            'summary': summary,
        }

        return render(request, 'notifications/notifications.html', context)

    @staticmethod
    @login_required
    def mark_notifications_read_view(request):
        """AJAX endpoint to mark notifications as read"""
        if request.method == 'POST':
            notification_ids = request.POST.getlist('notification_ids[]')

            if notification_ids:
                updated_count = NotificationsController.mark_notifications_read(request.user, notification_ids)
            else:
                updated_count = NotificationsController.mark_notifications_read(request.user)

            return JsonResponse({'success': True, 'updated_count': updated_count})

        return JsonResponse({'success': False}, status=400)
