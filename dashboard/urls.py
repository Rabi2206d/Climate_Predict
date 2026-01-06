from django.urls import path
from .controllers import DashboardController

app_name = 'dashboard'

urlpatterns = [
    path('', DashboardController.dashboard_view, name='dashboard'),
]
