"""
Management command to check alert thresholds and create notifications.
This can be run periodically via cron or task scheduler.
"""
from django.core.management.base import BaseCommand
from notifications.controllers import NotificationsController


class Command(BaseCommand):
    help = 'Check all active alert thresholds and create notifications if triggered'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output',
        )

    def handle(self, *args, **options):
        verbose = options['verbose']
        
        if verbose:
            self.stdout.write(self.style.SUCCESS('Starting alert threshold check...'))

        try:
            new_notifications = NotificationsController.check_alert_thresholds()
            
            if new_notifications:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully created {len(new_notifications)} new notification(s).'
                    )
                )
                if verbose:
                    for notification in new_notifications:
                        self.stdout.write(
                            f'  - {notification.title} (Priority: {notification.priority})'
                        )
            else:
                if verbose:
                    self.stdout.write(self.style.SUCCESS('No alert thresholds triggered.'))
                    
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error checking alert thresholds: {str(e)}')
            )
            raise

