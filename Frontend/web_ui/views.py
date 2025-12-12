from django.shortcuts import render
from django.conf import settings

def index(request):
    return render(request, 'index.html', {
        'api_base': getattr(settings, 'API_BASE_URL', 'http://127.0.0.1:8000')
    })
