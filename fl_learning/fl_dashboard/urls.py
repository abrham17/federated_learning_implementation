
from django.urls import path
from . import views
from django.contrib import admin
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.start_simulation, name='start_simulation'),
    path('dashoboard/' , views.dashboard, name='dashboard'),
]