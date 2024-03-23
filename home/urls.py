from django.urls import path
from home import views
from home.views import about
urlpatterns = [
    path('', views.index, name= 'home'),
    path('detection/', views.detection, name='detection'),
    path('about', about, name='about'),
    
]
