from django.urls import path
from .views import StockDataUniView, StockDataPredView, landing_page

urlpatterns = [
    path('', landing_page, name='landing'),  # Landing page route
    path('stock/uni/<str:ticker>/',
         StockDataUniView.as_view(), name='stock-data-uni'),
    path('stock/pred/', StockDataPredView.as_view(), name='select-stock'),
]
