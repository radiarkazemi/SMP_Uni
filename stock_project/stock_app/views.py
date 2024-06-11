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
from django.core.mail import send_mail
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import ContactForm
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import redirect
from .forms import RegisterForm


class StockDataUniView(View):
    def get(self, request, ticker):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['SMP_uni']
        collection = db[ticker]

        data = list(collection.find())
        if not data:
            return render(request, 'stock_app/stock_data_uni.html', {'error': 'No data found for this ticker in SMP_uni'})

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

        context = {
            'ticker': ticker,
            'stock_data': df.to_dict('records'),
            'stock_data_last_10_days': df.tail(10).to_dict('records')
        }

        # Create a line chart using Plotly
        line_chart = go.Figure(data=[go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price'
        )])
        line_chart.update_layout(title=f'Close Price for {ticker}',
                                 xaxis_title='Date',
                                 yaxis_title='Close Price')

        context['line_chart'] = plot(
            line_chart, output_type='div', include_plotlyjs=True)

        # Create a dot chart using Plotly
        dot_chart = go.Figure(data=[go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='markers',
            name='Close Price'
        )])
        dot_chart.update_layout(title=f'Close Price (Dot Chart) for {ticker}',
                                xaxis_title='Date',
                                yaxis_title='Close Price')

        context['dot_chart'] = plot(
            dot_chart, output_type='div', include_plotlyjs=True)

        return render(request, 'stock_app/stock_data_uni.html', context)


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
            line_chart = go.Figure(data=[go.Scatter(
                x=df['Date'],
                y=df['stock_value'],
                mode='lines',
                name='Stock Value'
            )])
            line_chart.update_layout(title=f'Stock Value for {context["ticker"]}',
                                     xaxis_title='Date',
                                     yaxis_title='Stock Value')

            context['line_chart'] = plot(
                line_chart, output_type='div', include_plotlyjs=True)

            # Create a dot chart using Plotly
            dot_chart = go.Figure(data=[go.Scatter(
                x=df['Date'],
                y=df['stock_value'],
                mode='markers',
                name='Stock Value'
            )])
            dot_chart.update_layout(title=f'Stock Value (Dot Chart) for {context["ticker"]}',
                                    xaxis_title='Date',
                                    yaxis_title='Stock Value')

            context['dot_chart'] = plot(
                dot_chart, output_type='div', include_plotlyjs=True)

        return render(request, 'stock_app/stock_data_pred.html', context)


def landing_page(request):
    return render(request, 'stock_app/landing.html')


def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']
            send_mail(
                f'Message from {name}',
                message,
                email,
                ['your_email@example.com'],
                fail_silently=False,
            )
            return HttpResponseRedirect(reverse('contact-success'))
    else:
        form = ContactForm()

    return render(request, 'stock_app/contact.html', {'form': form})


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('landing')
    else:
        form = RegisterForm()
    return render(request, 'stock_app/register.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('landing')
    else:
        form = AuthenticationForm()
    return render(request, 'stock_app/login.html', {'form': form})
