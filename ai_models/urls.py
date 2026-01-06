from django.urls import path
from .controllers import AIModelController

app_name = 'ai_models'

urlpatterns = [
    path('', AIModelController.ai_analysis_view, name='ai_dashboard'),
    path('analyze/', AIModelController.run_climate_analysis, name='run_analysis'),
    path('predictions/', AIModelController.predictions_history_view, name='predictions_history'),
    path('performance/', AIModelController.model_performance_view, name='model_performance'),
    path('train/', AIModelController.train_models_view, name='train_models'),
]
