from django.urls import path
from .controllers import NotificationsController

app_name = 'notifications'

urlpatterns = [
    path('mark-read/', NotificationsController.mark_notifications_read_view, name='mark_read'),
]
