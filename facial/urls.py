from django.urls import path
from . import views

urlpatterns = [
    path('facial-analysis/', views.facial_analysis, name='facial_analysis'),
]