import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid

# load JSON
def loadJSON(filepath):
    with open(filepath) as file:
        return json.load(file)
    
def removing_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = pd.read_csv('../data/raw/duplicates.csv')
    for index, row in duplicates.iterrows():
        id = row['id']
        category = row['category']
        df = df[~((df['id'] == id) & (df['category'] == category))]
    return df

def capping_time_series(df: pd.DataFrame) -> pd.DataFrame:
    token_info = df.groupby('id')['name'].value_counts()
    mean = np.floor(token_info.mean()).astype(int)
    ids_to_keep = token_info[token_info > mean].index.get_level_values(0).unique()
    filtered_meme = df[df['id'].isin(ids_to_keep)]
    first_timestamps = filtered_meme.groupby('id').timestamp.min()
    last_timestamps = filtered_meme.groupby('id').timestamp.max()
    max_first_timestamp = first_timestamps.max()
    min_last_timestamp = last_timestamps.min()
    return filtered_meme[filtered_meme['timestamp'] <= min_last_timestamp]

def train_test_split(df: pd.DataFrame, test_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_timestamp = df['timestamp'].max()
    test_start = last_timestamp - timedelta(days=test_days)
    train_df = df[df['timestamp'] < test_start]
    test_df = df[df['timestamp'] >= test_start]
    test_df = test_df.groupby('id').head(test_days)
    return train_df, test_df

def timestamp_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def scaling(df: pd.DataFrame):
    series_scaler = StandardScaler()
    exog_scaler = StandardScaler()
    halving_dates = [pd.to_datetime('2012-11-28'), pd.to_datetime('2016-07-09'), pd.to_datetime('2020-05-11'), pd.to_datetime('2024-04-20'), pd.to_datetime('2028-03-28')]
    df['days_until_halving'] = df['timestamp'].apply(lambda x: min([(halving - x).days for halving in halving_dates if halving > x]) if any(halving > x for halving in halving_dates) else 0).astype('float64')
    df['close'] = series_scaler.fit_transform(df[['close']])
    df['days_until_halving'] = exog_scaler.fit_transform(df[['days_until_halving']])
    return df, series_scaler, exog_scaler

def plot_time_series(df: pd.DataFrame, n: int) -> None:
    ids = df['id'].unique()
    for i in range(n):
        id = ids[i]
        token = df[df['id'] == id]
        plt.figure(figsize=(15, 5))
        plt.plot(token['timestamp'], token['close'])
        plt.title(f'Token ID: {id}')
        plt.show()

def create_series_exog(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_copy = df.copy()
    data_copy['seconds'] = data_copy['timestamp'].apply(lambda x: x.timestamp())
    
    day = 60 * 60 * 24
    week = day * 7
    month = week * 4

    data_copy['week_sin'] = np.sin(data_copy['seconds'] * (2 * np.pi / week))
    data_copy['week_cos'] = np.cos(data_copy['seconds'] * (2 * np.pi / week))
    data_copy['month_sin'] = np.sin(data_copy['seconds'] * (2 * np.pi / month))
    data_copy['month_cos'] = np.cos(data_copy['seconds'] * (2 * np.pi / month))

    series = data_copy[['timestamp', 'id', 'close']]
    exog = data_copy[['timestamp', 'id', 'days_until_halving', 'week_sin', 'week_cos', 'month_sin', 'month_cos']]
    return series, exog

def create_future_exog(df: pd.DataFrame, exog_scaler: StandardScaler, days: int = 60) -> pd.DataFrame:
    last_timestamp = df['timestamp'].max()
    future_dates = pd.date_range(start=last_timestamp + timedelta(days=1), periods=days, freq='D')
    
    future_data = pd.DataFrame({'timestamp': future_dates})
    future_data['seconds'] = future_data['timestamp'].apply(lambda x: x.timestamp())

    day = 60 * 60 * 24
    week = day * 7
    month = week * 4

    future_data['week_sin'] = np.sin(future_data['seconds'] * (2 * np.pi / week))
    future_data['week_cos'] = np.cos(future_data['seconds'] * (2 * np.pi / week))
    future_data['month_sin'] = np.sin(future_data['seconds'] * (2 * np.pi / month))
    future_data['month_cos'] = np.cos(future_data['seconds'] * (2 * np.pi / month))

    future_data['id'] = df['id'].iloc[0]

    halving_dates = [pd.to_datetime('2012-11-28'), pd.to_datetime('2016-07-09'), pd.to_datetime('2020-05-11'), pd.to_datetime('2024-04-20'), pd.to_datetime('2028-03-28')]
    future_data['days_until_halving'] = future_data['timestamp'].apply(lambda x: min([(halving - x).days for halving in halving_dates if halving > x]) if any(halving > x for halving in halving_dates) else 0).astype('float64')
    

    exog_future = future_data[['timestamp', 'id', 'days_until_halving', 'week_sin', 'week_cos', 'month_sin', 'month_cos']]
    exog_future.loc[:, 'days_until_halving'] = exog_scaler.transform(exog_future[['days_until_halving']])
    return exog_future

def create_all_future_exog(df: pd.DataFrame, exog_scaler: StandardScaler,days: int = 7) -> pd.DataFrame:
    future_exog_list = []
    
    for id, group in df.groupby('id'):
        future_exog = create_future_exog(group, exog_scaler,days)
        future_exog_list.append(future_exog)
    
    all_future_exog = pd.concat(future_exog_list, ignore_index=True)
    return all_future_exog

def create_dictionaries(series_df: pd.DataFrame, exog_df: pd.DataFrame, future_exog_df: pd.DataFrame) -> tuple[series_long_to_dict, exog_long_to_dict, exog_long_to_dict]:
    series_dict = series_long_to_dict(
                 data=series_df, 
                 series_id='id', 
                 index='timestamp', 
                 values= 'close', 
                 freq='D')

    exog_dict = exog_long_to_dict(
        data      = exog_df,
        series_id = 'id',
        index     = 'timestamp',
        freq      = 'D'
    )

    future_exog_dict = exog_long_to_dict(
        data      = future_exog_df,
        series_id = 'id',
        index     = 'timestamp',
        freq      = 'D'
    )

    return series_dict, exog_dict, future_exog_dict

def train_forecaster(series_dict: dict, exog_dict: exog_long_to_dict) -> ForecasterAutoregMultiSeries:
    regressor = LGBMRegressor(random_state=123, max_depth=5)
    print(regressor.get_params())
    forecaster = ForecasterAutoregMultiSeries(
                    regressor          = regressor,
                    lags               = 10,
                    encoding           = "ordinal",
                    dropna_from_series = False
                )
    
    forecaster.fit(series=series_dict, exog=exog_dict,suppress_warnings=True)
    
    return forecaster

def train_best_forecaster(series_dict: dict, exog_dict: dict, future_exog_dict: dict, test_data: pd.DataFrame, future_days: int) -> ForecasterAutoregMultiSeries:
    # Define the parameter grid for LGBMRegressor
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 500],
        'max_depth': [5],
        'num_leaves': [31, 50],
        'min_child_samples': [10, 20],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the LGBMRegressor
    lgbm = LGBMRegressor(random_state=123)

    # Keep track of the best forecaster based on MAPE
    best_forecaster = None
    best_mape = float('inf')
    best_params = None
    error_dict = {}

    # Perform grid search manually
    for params in ParameterGrid(param_grid):
        print(f"Training model with parameters: {params}")
        lgbm.set_params(**params)
        
        # Initialize forecaster
        forecaster = ForecasterAutoregMultiSeries(
            regressor=lgbm,
            lags=10,
            encoding="ordinal",
            dropna_from_series=False
        )
        
        # Train the forecaster
        forecaster.fit(series=series_dict, exog=exog_dict, suppress_warnings=True)
        
        # Make future predictions
        predictions = forecaster.predict(steps=future_days, exog=future_exog_dict)
        
        # Calculate errors (MSE, MAE, MAPE) on the test data
        for token_id in test_data['id'].unique():
            test_token_data = test_data[test_data['id'] == token_id].set_index('timestamp')['close']
            future_dates = test_token_data.index[:future_days]
            
            # Extract predictions for the token
            if token_id in predictions.columns:
                predicted_values = predictions[token_id].values[:future_days]
            else:
                continue  # Skip if no predictions available

            # Calculate errors for the overlap in future dates
            test_overlap = test_token_data.loc[future_dates]
            predictions_overlap = predicted_values[:len(test_overlap)]
            
            # Ensure there is no empty data
            if len(test_overlap) == 0:
                continue
            
            mse = mean_squared_error(test_overlap, predictions_overlap)
            mae = mean_absolute_error(test_overlap, predictions_overlap)
            mape = np.mean(np.abs((test_overlap - predictions_overlap) / test_overlap)) * 100
            
            # Store the errors for this token
            error_dict[token_id] = {'MSE': mse, 'MAE': mae, 'MAPE': mape}

            # Update the best forecaster based on MAPE
            if mape < best_mape:
                best_mape = mape
                best_forecaster = forecaster
                best_params = params

    # Print the best parameters
    print(f"Best Parameters: {best_params}")
    print(f"Best MAPE: {best_mape}")

    return best_forecaster

def predict_X_days(days_to_predict: int, forecaster: ForecasterAutoregMultiSeries, future_exog_dict: exog_long_to_dict) -> pd.DataFrame:
    return forecaster.predict(steps=days_to_predict, exog=future_exog_dict, suppress_warnings=True)

def plot_predictions(train_data: pd.DataFrame, predictions_x_days: pd.DataFrame, test_data: pd.DataFrame, last_data_points=-1, max_coins=None):
    tokens = test_data['id'].unique()  # Get unique tokens from test_data
    error_dict = {}

    if max_coins:
        tokens = tokens[:max_coins]

    for token_id in tokens:
        # Filter the historical (train) data and test data by token
        historical_data_token = train_data[train_data['id'] == token_id]
        test_data_token = test_data[test_data['id'] == token_id]
        
        if historical_data_token.empty or test_data_token.empty:
            continue  # Skip if no data is available for the token
        
        historical_data_token = historical_data_token.set_index('timestamp')
        test_data_token = test_data_token.set_index('timestamp')
        
        token_name = test_data_token['name'].iloc[0]
        token_symbol = test_data_token['symbol'].iloc[0]
        historical_data_token = historical_data_token['close']  # Extract only the 'close' column

        # Determine the number of steps ahead (based on the prediction length)
        steps_ahead = len(predictions_x_days)

        # Create a timeline for the predictions (future dates after the last historical data point)
        last_date = historical_data_token.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps_ahead + 1, freq='D')[1:]  # Skip the last date

        # Get the predicted values for the token
        if token_id in predictions_x_days.columns:
            predictions = pd.Series(predictions_x_days[token_id], index=future_dates, name='Predictions')
            test_data_token = test_data_token['close']
        else:
            continue

        # Get only the last `last_data_points` from historical data
        if last_data_points > -1:
            historical_data_token = historical_data_token[-last_data_points:]

        # Ensure the test data and predictions overlap in time
        overlap_idx = test_data_token.index.intersection(predictions.index)
        predictions_overlap = predictions.loc[overlap_idx]
        test_data_overlap = test_data_token.loc[overlap_idx]

        # Calculate the errors
        mse = mean_squared_error(test_data_overlap, predictions_overlap)
        mae = mean_absolute_error(test_data_overlap, predictions_overlap)
        mape = np.mean(np.abs((test_data_overlap - predictions_overlap) / test_data_overlap)) * 100

        # Store errors for the token
        error_dict[token_id] = {'MSE': mse, 'MAE': mae, 'MAPE': mape}

        # Plot the data
        plt.figure(figsize=(20, 5))
        if last_data_points != 0:
            plt.plot(historical_data_token.index, historical_data_token, label='Historical')
        plt.plot(predictions.index, predictions, label='Predictions', color='purple')
        plt.plot(test_data_token.index, test_data_token, label='Test Data', color='green')
        plt.axvline(x=last_date, color='red', linestyle='--', label='Start of Predictions')
        plt.title(f'Token: {token_name} ({token_symbol}) - Historical Data and {steps_ahead}-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend(title=f'Errors:\nMSE: {mse:.10f}\nMAE: {mae:.10f}\nMAPE: {mape:.2f}%')
        plt.grid(True)

        # Display plot
        plt.show()

def compute_errors(train_data: pd.DataFrame, predictions_x_days: pd.DataFrame, test_data: pd.DataFrame):
    tokens = test_data['id'].unique()  # Get unique tokens from test_data
    error_dict = {}

    for token_id in tokens:
        # Filter the historical (train) data and test data by token
        historical_data_token = train_data[train_data['id'] == token_id]
        test_data_token = test_data[test_data['id'] == token_id]
        
        if historical_data_token.empty or test_data_token.empty:
            continue  # Skip if no data is available for the token
        
        historical_data_token = historical_data_token.set_index('timestamp')
        test_data_token = test_data_token.set_index('timestamp')
        
        historical_data_token = historical_data_token['close']  # Extract only the 'close' column

        # Determine the number of steps ahead (based on the prediction length)
        steps_ahead = len(predictions_x_days)

        # Create a timeline for the predictions (future dates after the last historical data point)
        last_date = historical_data_token.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps_ahead + 1, freq='D')[1:]  # Skip the last date

        # Get the predicted values for the token
        if token_id in predictions_x_days.columns:
            predictions = pd.Series(predictions_x_days[token_id], index=future_dates, name='Predictions')
            test_data_token = test_data_token['close']
        else:
            continue

        # Ensure the test data and predictions overlap in time
        overlap_idx = test_data_token.index.intersection(predictions.index)
        predictions_overlap = predictions.loc[overlap_idx]
        test_data_overlap = test_data_token.loc[overlap_idx]

        # Calculate the errors
        mse = mean_squared_error(test_data_overlap, predictions_overlap)
        mae = mean_absolute_error(test_data_overlap, predictions_overlap)
        mape = np.mean(np.abs((test_data_overlap - predictions_overlap) / test_data_overlap)) * 100

        # Store errors for the token
        error_dict[token_id] = {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'MAPE (%)': f'{mape:.2f}%'}
    
    error_df = pd.DataFrame.from_dict(error_dict, orient='index')
    error_df.reset_index(inplace=True)
    error_df.rename(columns={'index': 'Token ID'}, inplace=True)

    return error_df

def inverse_scaling(train_df: pd.DataFrame, test_df:pd.DataFrame, pred_df: pd.DataFrame, series_scaler: StandardScaler) -> pd.DataFrame:
    train = train_df.copy()
    test = test_df.copy()
    pred = pred_df.copy()
    train['close'] = series_scaler.inverse_transform(train[['close']])
    test['close'] = series_scaler.inverse_transform(test[['close']])
    for column in pred_df.columns:
        pred[column] = series_scaler.inverse_transform(pred[[column]])

    return train, test, pred

def preprocess(data: pd.DataFrame, days_to_predict: int = 7):
    data_cleaned = removing_duplicates(data)
    data_capped = capping_time_series(data_cleaned)
    data_datetime = timestamp_to_datetime(data_capped)
    data_datetime.reset_index(drop=True, inplace=True)
    data_final, series_scaler, exog_scaler = scaling(data_datetime)
    train_data, test_data = train_test_split(data_final)
    series, exog = create_series_exog(train_data)
    future_exog = create_all_future_exog(train_data, exog_scaler=exog_scaler, days=days_to_predict)
    series_dict, exog_dict, future_exog_dict = create_dictionaries(series, exog, future_exog)

    return train_data, test_data, series_dict, exog_dict, future_exog_dict, series_scaler, exog_scaler

def get_last_close_info(historical_df: pd.DataFrame, test_df: pd.DataFrame, pred_df: pd.DataFrame):
    tokens = list(pred_df.columns)
    close_performance = {}
    for token_id in tokens:
        historical_data_token = historical_df[historical_df['id'] == token_id]
        test_data_token = test_df[test_df['id'] == token_id]

        if historical_data_token.empty or test_data_token.empty:
            continue

        historical_data_token = historical_data_token.set_index('timestamp')
        test_data_token = test_data_token.set_index('timestamp')

        last_historical_close = historical_data_token['close'].iloc[-1]
        last_test_close = test_data_token['close'].iloc[-1]
        last_pred_close = pred_df[token_id].iloc[-1]

        close_performance[token_id] = {'last_close': last_historical_close, 'last_test_close': last_test_close, 'last_pred_close': last_pred_close}

    close_performance_df = pd.DataFrame.from_dict(close_performance, orient='index')
    close_performance_df.reset_index(inplace=True)
    close_performance_df.rename(columns={'index': 'Token ID'}, inplace=True)

    close_performance_df['real_difference'] = close_performance_df.apply(lambda x: x['last_test_close'] - x['last_close'], axis=1)
    close_performance_df['pred_difference'] = close_performance_df.apply(lambda x: x['last_pred_close'] - x['last_close'], axis=1)
    close_performance_df['test_pred_difference'] = close_performance_df.apply(lambda x: x['last_test_close'] - x['last_pred_close'], axis=1)
    close_performance_df['real_went_up'] = close_performance_df.apply(lambda x: 1 if x['real_difference'] > 0 else 0, axis=1).astype(np.float64)
    close_performance_df['pred_went_up'] = close_performance_df.apply(lambda x: 1 if x['pred_difference'] > 0 else 0, axis=1).astype(np.float64)

    return close_performance_df