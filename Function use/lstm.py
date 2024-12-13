# -*- coding: utf-8 -*-
"""modeling_cnn_lstm_wuthout_docs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16HreQfKUOUM2w8O9Ab1ZygX16J-FYaOW
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from tensorflow.keras.utils import plot_model


# LSTM fit and forecast alpha version
def lstm_fit_and_forecast(data_var: pd.DataFrame, seq_length, n_steps, ratio_split,
                     output_file: str):

    # Step : Prepare data for LSTM and create train/test splits
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    gold_scaled = scaler.fit_transform(data_var[['USD']])

    # Define sequence length and split into training and testing sets
    seq_length = seq_length  # Use 12 months of past data to predict the next month

    X, y = create_sequences(gold_scaled, seq_length)
    train_size = int(ratio_split * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Reshape data for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

    # Step : Function for multi-step forecasting (1-6 steps ahead)
    def multi_step_forecast(model, X_input, n_steps):
        predictions = []
        input_seq = X_input.copy()

        for step in range(n_steps):
            pred = model.predict(input_seq.reshape(1, seq_length, 1))[0]
            predictions.append(pred)
            input_seq = np.append(input_seq[1:], pred)  # Shift the window forward

        return np.array(predictions)

    # Step : Generate 1 to 6 step ahead forecasts and save in DataFrame
    n_steps = n_steps
    forecast_results = pd.DataFrame()
    y_real = scaler.inverse_transform(X_test[:, -1].reshape(-1, 1)).flatten()
    forecast_results['y_real'] = y_real  # Add the true values as 'y_real'

    for i in range(1, n_steps + 1):
        y_pred = []
        for j in range(len(X_test)):
            input_seq = X_test[j]
            forecast = multi_step_forecast(model, input_seq, i)
            y_pred.append(forecast[-1])  # Take the ith step prediction
        forecast_results[f'{i}_step'] = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

    # Save forecast results to Excel
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.to_excel(output_file, index=False)

    # Step : Plot test vs predicted values for each step
    plt.figure(figsize=(12, 8))

    # Get the true values for the test set
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual values
    plt.plot(data_var['temps'][-len(y_test):], y_test_rescaled, label='Actual', color='blue')

    # Plot predicted values for each step
    for i in range(1, n_steps + 1):
        plt.plot(data_var['temps'][-len(y_test):], forecast_results[f'{i}_step'], label=f'{i}-Step Forecast')

    plt.title('Test vs Predicted Values (Multi-step Forecasting)')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.grid(True)
    plt.show()

#
def lstm_tune_hyperparameters(data, n_steps, dataset_name, sheet_name, max_epochs, factor):
    # Create sequences
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            end_ix = i + n_steps
            if end_ix + 1 < len(data):
                X.append(data[i:end_ix])
                y.append(data[end_ix])
        return np.array(X), np.array(y)

    # Build model function
    def build_lstm_model(hp):
        model = Sequential()
        # LSTM layer with tunable number of units
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                      activation='relu',
                      input_shape=(X_train.shape[1], 1)))
        # Output layer
        model.add(Dense(1))
        # Compile the model with tunable learning rate
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                      loss='mse')
        return model

    # Prepare data
    X, y = create_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    # Initialize the Keras Tuner with RandomSearch
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=5,  # You can increase this for a more exhaustive search
        executions_per_trial=1,
        directory='lstm_tuning',
        project_name='lstm')

    # Search for the best hyperparameters
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    # Save the best model in Keras format
    best_model.save(f'best_model_{dataset_name}_{sheet_name}.keras')

    # Save the architecture of the model as a JPEG
    plot_model(best_model, to_file=f'model_architecture_{dataset_name}_{sheet_name}.png', show_shapes=True, show_layer_names=True)

    # Save the best hyperparameters as a JSON file
    with open(f'best_hyperparameters_{dataset_name}_{sheet_name}.json', 'w') as f:
        json.dump(best_hyperparameters.values, f)

    return best_hyperparameters, best_model