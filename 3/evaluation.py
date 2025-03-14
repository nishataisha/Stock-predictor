#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


output_dir = os.path.join('Workshop 4', 'result', 'randomForest_plots')
os.makedirs(output_dir, exist_ok=True)

from stockprice.lstm import lstm_train_predict
from stockprice.randomForest import randomForest_train_predict
from stockprice.svr import svr_train_predict

data = pd.read_csv('C:\\Users\\Fareesa Meraj\\AI\\stockprice\\stocks_data (1).csv')

df = pd.DataFrame(data)


def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"{model} evaluation for {stock}:")

    mse = mean_squared_error(Y_test, Y_predictions)
    r2 = r2_score(Y_test, Y_predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return r2, mse


stocks = data['Name'].unique() #Getting all different stocks in the Data
features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
target = 'close'


results = {'lstm_r2' : [], 'lstm_mse' : [],
           'randomForest_r2' : [], 'randomForest_mse' : [],
           'svr_r2' : [], 'svr_mse' : []} 

for stock in stocks: 

    
    stock_data = data[data['Name'] == stock] 

    X = stock_data[features] 
    y = stock_data[target] 
    dates = stock_data['date'] 

    X_train, X_test, Y_train, Y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state= 42)

    
    X_test_sorted = X_test.sort_index()
    y_test_sorted = Y_test.sort_index()
    dates_test_sorted = pd.to_datetime(dates_test.sort_index())


    Y_predictions = lstm_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('LSTM', stock, y_test_sorted, Y_predictions)

    results['lstm_r2'].append(r2)
    results['lstm_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='LSTM Predictions', color='purple')
    plt.title(f'{stock} - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join(output_dir, f'{stock}_randomForest.png'))
    plt.close()

    
    Y_predictions = randomForest_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('Random Forest', stock, y_test_sorted, Y_predictions)

    results['randomForest_r2'].append(r2)
    results['randomForest_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='Random Forest Predictions', color='purple')
    plt.title(f'{stock} - Random Forest')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/randomForest_plots', f'{stock}_randomForest.png'))
    plt.close()

    
    Y_predictions = svr_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('SVR', stock, y_test_sorted, Y_predictions)

    results['svr_r2'].append(r2)
    results['svr_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='SVR Predictions', color='purple')
    plt.title(f'{stock} - SVR')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    # For example, to save in the "stockprice" directory
    plt.savefig(os.path.join('C:\\Users\\Fareesa Meraj\\AI\\stockprice', f'{stock}_lstm.png'))
    plt.close()

print("Results:")

# LSTM Average Results 
lstm_r2 = np.mean(results['lstm_r2'])
lstm_mse = np.mean(results['lstm_mse'])
print(f"LSTM Average R2 Score: {lstm_r2:.4f}")
print(f"LSTM Average Mean Squared Error: {lstm_mse:.4f}")


randomForest_r2 = np.mean(results['randomForest_r2'])
randomForest_mse = np.mean(results['randomForest_mse'])
print(f"Random Forest Average R2 Score: {randomForest_r2:.4f}")
print(f"Random Forest Average Mean Squared Error: {randomForest_mse:.4f}")

svr_r2 = np.mean(results['svr_r2'])
svr_mse = np.mean(results['svr_mse'])
print(f"SVR Average R2 Score: {svr_r2:.4f}")
print(f"SVR Average Mean Squared Error: {svr_mse:.4f}")