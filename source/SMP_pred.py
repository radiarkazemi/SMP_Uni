from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import logging
import json
import jdatetime
import datetime
import requests
import os


# Function to Update date and convert it to jdatetime and also remove the holidays
def updated_date(dates):
    holidays = ['1403-02-31', '1403-03-02']
    holidays = [jdatetime.date(
        *map(int, holiday.split('-'))).togregorian() for holiday in holidays]

    strdate = []
    new_dates = []

    for date in dates:
        date = date.date()  # Convert pandas Timestamp to datetime.date
        if date.weekday() in [3, 4]:  # Thursday: 3, Friday: 4
            new_date = date + timedelta(days=2)
            while new_date in new_dates or new_date in holidays:
                new_date += timedelta(days=1)
            new_dates.append(new_date)
        else:
            if date not in new_dates and date not in holidays:
                new_dates.append(date)

    max_date = max(new_dates)
    for i in range(1, 3):
        next_date = max_date + timedelta(days=i)
        while next_date in holidays:
            next_date += timedelta(days=1)
        new_dates.append(next_date)

    for item in new_dates:
        if item not in holidays:
            jalali_date = jdatetime.date.fromgregorian(date=item)
            strdate.append(jalali_date.strftime('%Y-%m-%d'))

    return strdate


def fetch_data_from_mongodb(ticker, db_name='SMP_uni', collection_field='data'):
    """
    Fetches stock data for a given ticker from MongoDB.

    Args:
        ticker (str): Stock ticker.
        db_name (str): The name of the MongoDB database.
        collection_field (str): The field name in MongoDB that contains the data.

    Returns:
        pd.DataFrame: DataFrame containing the stock data for the ticker.
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[ticker]

    # Fetch data from MongoDB
    cursor = collection.find()
    data = list(cursor)

    if not data:
        print(f"No data found for ticker {ticker}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract nested data
    data_df = pd.json_normalize(df[collection_field])

    # Combine with main DataFrame
    df = pd.concat([df.drop(columns=[collection_field]), data_df], axis=1)

    # Drop Time.$date if it exists
    if 'Time.$date' in df.columns:
        df = df.drop(columns=['Time.$date'])

    if 'company_name' in df.columns:
        df = df.drop(columns=['company_name'])

    if 'ticker' in df.columns:
        df = df.drop(columns=['ticker'])

    # Rename _id to Date and set it as the index
    df.rename(columns={'_id': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)

    return df


def preprocess_data(ticker):
    # if u call preprocess_date() ---> it will set daytogoback=26
    # if u call preprocess_date(daytogoback=n) ---> it will set daytogoback=n
    # Function to calculate RSI
    def calc_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Function to calculate MACD
    def calc_macd(df, short_window=12, long_window=26, signal_window=9):
        df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(
            span=signal_window, adjust=False).mean()
        return df
    df = fetch_data_from_mongodb(ticker)
    # Converting relavent columns to float
    df[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ] = df[[
        'Open', 'High', 'Low', 'Close', 'Volume']
    ].astype(float)
    df.sort_index(ascending=True, inplace=True)

    # Create 3-Day EMA
    df['EMA_3'] = df['Close'].ewm(span=3, adjust=False).mean()

    # Create 5-Day EMA
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

    # Calculate EMA percentage change
    df['EMA_pct_change'] = ((df['EMA_3'] - df['EMA_5']) / 100) * 100

    # Create 14-Day RSI
    df['RSI'] = calc_rsi(df['Close'])

    # Create 9-Day MACD Signal
    df = calc_macd(df)

    df.dropna(inplace=True)

    if df.isnull().values.any():
        print('Missing Values found. Filling missing valued...')
        df = df.fillna(method='ffill').fillna(method='bfill')

    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)

    return scaled_df, scaler


# Function to predinct the close prices for the next 'num_days' based on the last 'window_size' days
def predict_next_days(model, df, num_days=5, window_size=5):
    filtered_df = df.filter(
        ['Close', 'EMA_pct_change', 'RSI', 'MACD', 'MACD_Signal'])

    # Convert the last date to a datetime object
    last_date = pd.to_datetime(filtered_df.index[-2])
    predicted_dates = []
    predicted_prices = []

    # Ensure it is a 1D array
    last_window = filtered_df.iloc[-window_size:, :].values.flatten()
    for _ in range(num_days):
        input_data = last_window.reshape((1, window_size, 5))
        next_day_prediction = model.predict(input_data, verbose=0)[0][0]

        last_date += pd.Timedelta(days=1)
        predicted_dates.append(last_date)
        predicted_prices.append(next_day_prediction)

        last_window = np.append(last_window[1:], next_day_prediction)

    return predicted_dates, predicted_prices
# Inverse transform the predicted and actual values to original scale


def inverse_transform_predictions(scaler, scaled_df, prediction_df, predicted_df):
    prediction_df_full = scaled_df.copy()

    # Ensure the length of prediction_df matched the length of the DataFrame
    close_values = prediction_df['Close'].values
    if len(close_values) != len(prediction_df_full):
        raise ValueError('Length of values does not match length of DataFrame')

    prediction_df_full.loc[:, 'Close'] = close_values.astype(float)
    prediction_df_full = scaler.inverse_transform(prediction_df_full)
    pre['Close'] = prediction_df_full[:,
                                      scaled_df.columns.get_loc('Close')]

    temp_df = pd.DataFrame(
        np.zeros((len(predicted_df), scaled_df.shape[1])), columns=scaled_df.columns
    )
    temp_df['Close'] = predicted_df['Predicted Close']
    predicted_df['Predicted Close'] = scaler.inverse_transform(
        temp_df
    )[:, scaled_df.columns.get_loc('Close')]

    return predicted_df


# Adjust the predicted closing price by the diffrence between the last actual close and first predicted close
def adjust_predictions(prediction_df, predicted_df):
    price_shift = prediction_df.iloc[-1]['Close'] - \
        predicted_df.iloc[0]['Predicted Close']
    predicted_df_shifted = predicted_df['Predicted Close'] + price_shift
    return predicted_df_shifted.to_list(), updated_date(predicted_df['Date'].to_list())


# Store the prediction result in MongoDB
def store_prediction_in_mongodb(dates, results, model_name):
    data = [{'Time': int(time.mktime(jdatetime.date.fromisoformat(date).togregorian().timetuple())),
            'jalali_date': date,
             'gregorian_date': jdatetime.date.fromisoformat(date).togregorian().isoformat(),
             'stock_value': stock}
            for date, stock in zip(dates, results)]

    client = MongoClient('mongodb://localhost:27017/')
    db = client['SMP_uni_pred']
    collection = db[model_name]

    today_date = datetime.datetime.today().strftime('%Y-%m-%d')

    existing_document = collection.find_one({'_id': today_date})

    if existing_document:
        collection.update_one({'_id': today_date}, {'$set': {'data': data}})
    else:
        collection.insert_one({'_id': today_date, 'data': data})


def get_h5_files(directory):
    h5_files = [file.replace('.h5', '') for file in os.listdir(
        directory) if file.endswith('.h5')]
    return h5_files


def main(model, scaler, prediction_df, scaled_df, df, model_name):
    predicted_dates, predicted_price = predict_next_days(
        model, prediction_df, num_days=5, window_size=5)

    predicted_df = pd.DataFrame(
        {'Date': predicted_dates, 'Predicted Close': predicted_price})
    predicted_df = inverse_transform_predictions(
        scaler, scaled_df, prediction_df, predicted_df)

    shifted_predictions, updated_date = adjust_predictions(
        prediction_df, predicted_df)

    store_prediction_in_mongodb(updated_date, shifted_predictions, model_name)

    return shifted_predictions, updated_date


def SMP(model_name):
    df, scaler = preprocess_data(stk)

    if df is None or scaler is None:
        return None, None, None

    model = load_model(f'Models/{model_name}.h5')

    return df, scaler, model


model_directory = 'Models'
files = get_h5_files(model_directory)
for model_name in tqdm(files):
    stk = model_name
    try:
        df, scaler, model = SMP(model_name)
        if df is not None and scaler is not None and model is not None:
            pre = df.copy()
            scaled_df = df.copy()

            shifted_predictions, updated_dates = main(
                model, scaler, pre, scaled_df, df, model_name)

            print(f'Stock Name: {model_name}')
            print(shifted_predictions)
            print(updated_dates)
    except Exception as e:
        print(f'Error processing {model_name}: {e}')
        continue
