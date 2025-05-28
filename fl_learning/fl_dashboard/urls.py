from django.urls import path
from . import views

urlpatterns = [
    path('start/', views.start_simulation, name='start_simulation'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('metrics/', views.get_metrics, name='get_metrics'),
    path('', views.dashboard, name='home'),
]