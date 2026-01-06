from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from .models import DataSource, DataUpload


class DataSourceForm(forms.ModelForm):
    """Form for creating and editing data sources"""

    class Meta:
        model = DataSource
        fields = [
            'name',
            'source_type',
            'location',
            'latitude',
            'longitude',
            'description'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'latitude': forms.NumberInput(attrs={'step': '0.0000001'}),
            'longitude': forms.NumberInput(attrs={'step': '0.0000001'}),
        }


class DataUploadForm(forms.ModelForm):
    """Form for uploading climate data files"""

    class Meta:
        model = DataUpload
        fields = ['file_path']
        widgets = {
            'file_path': forms.FileInput(attrs={'accept': '.csv,.json,.xml,.txt'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set filename when file is uploaded
        if self.instance and self.instance.file_path:
            self.instance.filename = self.instance.file_path.name
