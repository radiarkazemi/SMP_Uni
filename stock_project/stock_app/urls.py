from django.urls import path
from .views import StockDataUniView, StockDataPredView, landing_page, contact, register, user_login
from django.views.generic import TemplateView

urlpatterns = [
    path('', landing_page, name='landing'),  # Landing page route
    path('stock/uni/', StockDataUniView.as_view(), name='stock-data-uni'),
    path('stock/uni/<str:ticker>/', StockDataUniView.as_view(),
         name='stock-data-uni-ticker'),
    path('stock/pred/', StockDataPredView.as_view(), name='select-stock'),
    path('contact/', contact, name='contact'),
    path('contact/success/', TemplateView.as_view(
        template_name='stock_app/contact_success.html'), name='contact-success'),
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
]
