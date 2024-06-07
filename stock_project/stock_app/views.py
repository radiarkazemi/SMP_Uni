import pandas as pd
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views import View
from pymongo import MongoClient
from .serializers import StockDataUniSerializer, StockDataPredSerializer
from django.shortcuts import render
from .forms import StockTickerForm
import plotly.graph_objs as go
from plotly.offline import plot


class StockDataUniView(APIView):
    def get(self, request, ticker):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['SMP_uni']
        collection = db[ticker]

        data = list(collection.find())
        if not data:
            return Response({'error': 'No data found for this ticker in SMP_uni'}, status=404)

        df = pd.DataFrame(data)
        data_df = pd.json_normalize(df['data'])
        df = pd.concat([df.drop(columns=['data']), data_df], axis=1)

        columns_to_drop = ['Time.$date', 'company_name', 'ticker']
        df.drop(
            columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        df.rename(columns={'_id': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.reset_index(inplace=True)

        serialized_data = StockDataUniSerializer(
            df.to_dict('records'), many=True)
        return Response(serialized_data.data)


# Dictionary to map stock symbols to human-readable names
STOCK_NAME_MAPPING = {
    'AAPL': 'Apple Inc.',
    'GOOG': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    # Add more mappings as needed
}


class StockDataPredView(View):
    def get(self, request):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['SMP_uni_pred']

        # Fetch collection names and map them to human-readable stock names
        stock_choices = [(name, STOCK_NAME_MAPPING.get(name, name))
                         for name in db.list_collection_names()]

        form = StockTickerForm(stock_choices=stock_choices)
        context = {'form': form}
        return render(request, 'stock_app/stock_data_pred.html', context)

    def post(self, request):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['SMP_uni_pred']

        # Fetch collection names and map them to human-readable stock names
        stock_choices = [(name, STOCK_NAME_MAPPING.get(name, name))
                         for name in db.list_collection_names()]

        form = StockTickerForm(request.POST, stock_choices=stock_choices)
        context = {'form': form}

        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            collection = db[ticker]

            data = list(collection.find())
            if not data:
                context['error'] = f'No data found for ticker {ticker} in SMP_uni_pred'
                return render(request, 'stock_app/stock_data_pred.html', context)

            flattened_data = []
            for document in data:
                date = document.get('_id')
                for entry in document.get('data', []):
                    flattened_entry = {
                        'Date': date,
                        'Time': entry.get('Time'),
                        'jalali_date': entry.get('jalali_date'),
                        'gregorian_date': entry.get('gregorian_date'),
                        'stock_value': entry.get('stock_value')
                    }
                    flattened_data.append(flattened_entry)

            df = pd.DataFrame(flattened_data)
            df['Date'] = pd.to_datetime(df['gregorian_date'])
            df.set_index('Date', inplace=True)
            df.reset_index(inplace=True)

            context['ticker'] = STOCK_NAME_MAPPING.get(ticker, ticker)
            context['stock_data'] = df.to_dict('records')

            # Create a line chart using Plotly
            fig = go.Figure(data=[go.Scatter(
                x=df['Date'],
                y=df['stock_value'],
                mode='lines',
                name='Stock Value'
            )])
            fig.update_layout(title=f'Stock Value for {context["ticker"]}',
                              xaxis_title='Date',
                              yaxis_title='Stock Value')

            context['line_chart'] = plot(
                fig, output_type='div', include_plotlyjs=True)

        return render(request, 'stock_app/stock_data_pred.html', context)


def landing_page(request):
    return render(request, 'landing.html')
