import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

# Load historical data for Amazon
amazon_data = pd.read_csv('amazon.csv')

# Convert 'Date' column to datetime
amazon_data['Date'] = pd.to_datetime(amazon_data['Date'])

# Use historical data to train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(amazon_data[['Open', 'High', 'Low', 'Close', 'Volume']], amazon_data['Adj Close'])

# Fetch real-time data for Amazon
ticker_symbol = 'AMZN'

# Create empty lists to store real-time data
amazon_timestamps = []
amazon_prices = []
amazon_predicted_prices = []

# Create the plot
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')

# Function to update the plot
def update_amazon_plot():
    plt.plot(amazon_timestamps, amazon_prices, label='Real-Time Price', color='purple')
    plt.plot(amazon_timestamps, amazon_predicted_prices, label='Predicted Price', linestyle='--', color='orange')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')
    plt.pause(0.01)

# Function to predict end price for the day
def predict_end_of_day_price(today, realtime_data):
    # Get historical data for the same day of the week
    historical_data_same_weekday = amazon_data[amazon_data['Date'].dt.weekday == today.weekday()]
    average_price_change_same_day = historical_data_same_weekday['Close'].diff().mean()
    return realtime_data['Close'].iloc[-1] + average_price_change_same_day

# Function to predict end price for the week
def predict_end_of_week_price(today, realtime_data):
    # Get historical data for the same weekday over several weeks
    historical_data_same_weekday = amazon_data[amazon_data['Date'].dt.weekday == today.weekday()]
    average_price_change_same_week = historical_data_same_weekday['Close'].diff(periods=5).mean()  # Change 5 to desired number of weeks
    return realtime_data['Close'].iloc[-1] + average_price_change_same_week

# Function to predict end price for the month
def predict_end_of_month_price(today, realtime_data):
    # Get historical data for the same month over several years
    historical_data_same_month = amazon_data[amazon_data['Date'].dt.month == today.month]
    average_price_change_same_month = historical_data_same_month['Close'].diff(periods=12).mean()  # Change 12 to desired number of months
    return realtime_data['Close'].iloc[-1] + average_price_change_same_month

# Infinite loop to continuously update the plot with real-time data
while True:
    try:
        # Fetch real-time data
        today = datetime.now().date()
        amazon_stock = yf.Ticker(ticker_symbol)
        amazon_realtime_data = amazon_stock.history(period='1d')

        # Extract the latest timestamp and price
        amazon_latest_timestamp = datetime.now()
        amazon_latest_price = amazon_realtime_data['Close'].iloc[-1]

        # Append data to lists
        amazon_timestamps.append(amazon_latest_timestamp)
        amazon_prices.append(amazon_latest_price)

        # Predict price using the Random Forest Regression model
        amazon_predicted_price = model.predict([[amazon_realtime_data['Open'].iloc[-1], amazon_realtime_data['High'].iloc[-1], 
                                                 amazon_realtime_data['Low'].iloc[-1], amazon_realtime_data['Close'].iloc[-1], 
                                                 amazon_realtime_data['Volume'].iloc[-1]]])[0]
        amazon_predicted_prices.append(amazon_predicted_price)

        # Update the plot
        plt.clf()  # Clear the current figure
        update_amazon_plot()

        # Wait for 10 seconds before fetching the next data
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
