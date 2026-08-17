from django.urls import path
from .views import index, analysis

urlpatterns = [
    path('', index),
    path('analysis/', analysis),
]
