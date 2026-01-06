from django.urls import path
from .controllers import DataController

app_name = 'data'

urlpatterns = [
    path('sources/', DataController.data_sources_view, name='sources'),
    path('sources/<int:source_id>/delete/', DataController.delete_data_source, name='delete_source'),
    path('sources/<int:source_id>/update/', DataController.update_data_source, name='update_source'),
    path('sources/<int:source_id>/toggle/', DataController.toggle_data_source_active, name='toggle_source'),
    path('sources/<int:source_id>/get/', DataController.get_data_source, name='get_source'),
    path('upload/', DataController.data_upload_view, name='upload'),
    path('analytics/', DataController.data_analytics_view, name='analytics'),
    path('analytics/detailed/', DataController.detailed_analytics_view, name='detailed_analytics'),
]