from django.urls import path
from .controllers import NotificationsController

app_name = 'notifications'

urlpatterns = [
    path('', NotificationsController.notifications_view, name='notifications'),
    path('mark-read/', NotificationsController.mark_notifications_read_view, name='mark_read'),
]
