from django.urls import path
from .controllers import AuthenticationController

app_name = 'authentication'

urlpatterns = [
    path('login/', AuthenticationController.login_view, name='login'),
    path('logout/', AuthenticationController.logout_view, name='logout'),
    path('profile/', AuthenticationController.profile_view, name='profile'),
    path('users/', AuthenticationController.user_list_view, name='user_list'),
    path('users/add/', AuthenticationController.add_user_view, name='add_user'),
    path('users/<int:user_id>/edit/', AuthenticationController.edit_user_view, name='edit_user'),
    path('users/<int:user_id>/delete/', AuthenticationController.delete_user_view, name='delete_user'),
]
