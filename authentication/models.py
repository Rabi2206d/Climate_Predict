from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

class User(AbstractUser):
    """Custom user model for Climate Agency with role-based access"""

    class Role(models.TextChoices):
        ADMIN = 'ADMIN', _('Administrator')
        ANALYST = 'ANALYST', _('Data Analyst')
        VIEWER = 'VIEWER', _('Data Viewer')

    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.VIEWER,
        help_text="User role for access control"
    )

    department = models.CharField(
        max_length=100,
        blank=True,
        help_text="User's department within the agency"
    )

    phone = models.CharField(
        max_length=20,
        blank=True,
        help_text="Contact phone number"
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Designates whether this user should be treated as active"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
