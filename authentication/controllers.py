"""
Authentication Controllers - Handle business logic for user authentication
"""
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from .models import User

class AuthenticationController:
    """Controller for authentication-related operations"""

    @staticmethod
    @csrf_protect
    def login_view(request):
        """Handle user login - Complete view logic"""
        if request.user.is_authenticated:
            return redirect('dashboard:dashboard')

        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')
                next_url = request.GET.get('next', 'dashboard:dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')

        return render(request, 'authentication/login.html')

    @staticmethod
    @csrf_protect
    def register_view(request):
        """Handle user registration"""
        if request.user.is_authenticated:
            return redirect('dashboard:dashboard')

        if request.method == 'POST':
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            
            # Basic validation
            if password != confirm_password:
                messages.error(request, 'Passwords do not match.')
                return render(request, 'authentication/register.html')
            
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists.')
                return render(request, 'authentication/register.html')
                
            if User.objects.filter(email=email).exists():
                messages.error(request, 'Email already registered.')
                return render(request, 'authentication/register.html')
            
            try:
                # Create user with default role VIEWER or ANALYST
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    role=User.Role.ANALYST  # Default role
                )
                
                messages.success(request, 'Account created successfully! Please sign in.')
                return redirect('authentication:login')
            except Exception as e:
                messages.error(request, f'Registration failed: {str(e)}')
                
        return render(request, 'authentication/register.html')

    @staticmethod
    @login_required
    def logout_view(request):
        """Handle user logout - Complete view logic"""
        logout(request)
        messages.info(request, 'You have been logged out.')
        return redirect('authentication:login')

    @staticmethod
    @login_required
    def profile_view(request):
        """Handle user profile - Complete view logic"""
        if request.method == 'POST':
            # Handle profile update without forms
            # Refresh user from database to ensure we have latest data
            user = User.objects.get(id=request.user.id)
            user.first_name = request.POST.get('first_name', '').strip()
            user.last_name = request.POST.get('last_name', '').strip()
            user.email = request.POST.get('email', '').strip()
            user.department = request.POST.get('department', '').strip()
            user.phone = request.POST.get('phone', '').strip()
            user.save()

            messages.success(request, 'Profile updated successfully!')
            return redirect('authentication:profile')

        # Display current profile data - refresh from database to ensure latest data
        user = User.objects.get(id=request.user.id)
        context = {
            'user': user,
        }
        return render(request, 'authentication/profile.html', context)

    @staticmethod
    @login_required
    def user_list_view(request):
        """List all users - Admin only"""
        if request.user.role != User.Role.ADMIN:
            messages.error(request, 'Access denied. Administrator privileges required.')
            return redirect('dashboard:dashboard')

        users = User.objects.all().order_by('-date_joined')
        context = {
            'users': users,
            'total_users': users.count(),
            'admin_count': users.filter(role=User.Role.ADMIN).count(),
            'analyst_count': users.filter(role=User.Role.ANALYST).count(),
            'viewer_count': users.filter(role=User.Role.VIEWER).count(),
        }
        return render(request, 'authentication/user_list.html', context)

    @staticmethod
    @login_required
    def add_user_view(request):
        """Add new user - Admin only"""
        if request.user.role != User.Role.ADMIN:
            messages.error(request, 'Access denied. Administrator privileges required.')
            return redirect('dashboard:dashboard')

        if request.method == 'POST':
            # Create user directly (no forms.py)
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password1')
            role = request.POST.get('role')
            first_name = request.POST.get('first_name', '')
            last_name = request.POST.get('last_name', '')
            department = request.POST.get('department', '')
            phone = request.POST.get('phone', '')

            try:
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                    role=role,
                    department=department,
                    phone=phone
                )
                messages.success(request, f'User {username} created successfully!')
                return redirect('authentication:user_list')
            except Exception as e:
                messages.error(request, f'Error creating user: {str(e)}')

        context = {
            'roles': User.Role.choices,
        }
        return render(request, 'authentication/add_user.html', context)

    @staticmethod
    @login_required
    def edit_user_view(request, user_id):
        """Edit user - Admin only"""
        if request.user.role != User.Role.ADMIN:
            messages.error(request, 'Access denied. Administrator privileges required.')
            return redirect('dashboard:dashboard')

        try:
            target_user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            messages.error(request, 'User not found.')
            return redirect('authentication:user_list')

        if request.method == 'POST':
            # Update user directly (no forms.py)
            target_user.first_name = request.POST.get('first_name', target_user.first_name)
            target_user.last_name = request.POST.get('last_name', target_user.last_name)
            target_user.email = request.POST.get('email', target_user.email)
            target_user.role = request.POST.get('role', target_user.role)
            target_user.department = request.POST.get('department', target_user.department)
            target_user.phone = request.POST.get('phone', target_user.phone)
            target_user.is_active = request.POST.get('is_active') == 'on'

            target_user.save()
            messages.success(request, f'User {target_user.username} updated successfully!')
            return redirect('authentication:user_list')

        context = {
            'edit_user': target_user,
            'roles': User.Role.choices,
        }
        return render(request, 'authentication/edit_user.html', context)

    @staticmethod
    @login_required
    def delete_user_view(request, user_id):
        """Delete user - Admin only"""
        if request.user.role != User.Role.ADMIN:
            messages.error(request, 'Access denied. Administrator privileges required.')
            return redirect('dashboard:dashboard')

        try:
            target_user = User.objects.get(id=user_id)
            if target_user == request.user:
                messages.error(request, 'Cannot delete your own account.')
                return redirect('authentication:user_list')

            username = target_user.username
            target_user.delete()
            messages.success(request, f'User {username} deleted successfully!')
        except User.DoesNotExist:
            messages.error(request, 'User not found.')

        return redirect('authentication:user_list')
