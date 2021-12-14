from django.urls import path,include
from . import views 
urlpatterns = [
    path('',views.index),
    path('login.html',views.login,name='login'),
    path('signup.html',views.signup,name='signup'),
    path('upload.html',views.uploadfiles,name='upload'),
    path('report1.html',views.report1,name='report1'),
    path('result1.html',views.result1,name='result1'),
    path('report2.html',views.report2,name='report2'),
    path('result2.html',views.result2,name='result2'),
    path('accounts/profile/',views.uploadfiles,name='upload'),
     path('analyze.html',views.analyze,name='analyze'),
]
