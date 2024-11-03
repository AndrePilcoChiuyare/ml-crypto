import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict
from datetime import timedelta
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from catboost import CatBoostRegressor
import json

class Model:
    def __init__(self) -> None:
        self.train_data = None
        self.series_dict = None
        self.exog_dict = None
        self.future_exog_dict = None
        self.series_scaler = None
        self.exog_scaler = None
        self.forecaster = None
        self.predictions = None
        self.og_train = None
        self.og_pred = None
        self.complete_time_series = None
        self.days_to_predict = None

    def loadJSON(self, filepath):
        with open(filepath) as file:
            return json.load(file)

    def load_complete_time_series(self, category: str):
        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')
        return self.loadJSON(f'../data/processed/{category}_complete_time_series.json')

    def train_predict(self, category: str, days_to_predict: int = 7, model: str = "catboost") -> None:
        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')
        data = pd.read_csv(f'../data/processed/{category}.csv')
        self.days_to_predict = days_to_predict
        self.preprocess(data, days_to_predict=self.days_to_predict)
        self.predict(model=model)

    def removing_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        duplicates = pd.read_csv('../data/raw/duplicates.csv')
        for index, row in duplicates.iterrows():
            id = row['id']
            category = row['category']
            df = df[~((df['id'] == id) & (df['category'] == category))]
        return df

    def capping_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        token_info = df.groupby('id')['name'].value_counts()
        mean = np.floor(token_info.mean()).astype(int)
        ids_to_keep = token_info[token_info > mean].index.get_level_values(0).unique()
        filtered_meme = df[df['id'].isin(ids_to_keep)]
        first_timestamps = filtered_meme.groupby('id').timestamp.min()
        last_timestamps = filtered_meme.groupby('id').timestamp.max()
        max_first_timestamp = first_timestamps.max()
        last_timestamp = last_timestamps.max()
        ids_to_keep2 = last_timestamps[last_timestamps == last_timestamp].index
        return filtered_meme[filtered_meme['id'].isin(ids_to_keep2)]

    def timestamp_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def scaling(self, df: pd.DataFrame):
        scalers = {}
        df_scaled = df.copy()

        # Escalado global para days_until_halving
        exog_scaler = StandardScaler()
        halving_dates = [pd.to_datetime('2012-11-28'), pd.to_datetime('2016-07-09'), pd.to_datetime('2020-05-11'), pd.to_datetime('2024-04-20'), pd.to_datetime('2028-03-28')]
        df_scaled['days_until_halving'] = df['timestamp'].apply(lambda x: min([(halving - x).days for halving in halving_dates if halving > x]) if any(halving > x for halving in halving_dates) else 0).astype('float64')
        df_scaled['days_until_halving'] = exog_scaler.fit_transform(df_scaled[['days_until_halving']])

        # Escalado específico para close por cada token
        for token_id in df['id'].unique():
            token_data = df[df['id'] == token_id]
            series_scaler = StandardScaler()
            
            # Escalar 'close' por cada token
            df_scaled.loc[df['id'] == token_id, 'close'] = series_scaler.fit_transform(token_data[['close']])
            
            # Guardar el scaler para usarlo en la inversión del escalado
            scalers[token_id] = series_scaler

        return df_scaled, scalers, exog_scaler

    def create_series_exog(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def create_future_exog(self, df: pd.DataFrame, exog_scaler: StandardScaler, days: int = 60) -> pd.DataFrame:
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

    def create_all_future_exog(self, df: pd.DataFrame, exog_scaler: StandardScaler,days: int = 7) -> pd.DataFrame:
        future_exog_list = []
        
        for id, group in df.groupby('id'):
            future_exog = self.create_future_exog(group, exog_scaler,days)
            future_exog_list.append(future_exog)
        
        all_future_exog = pd.concat(future_exog_list, ignore_index=True)
        return all_future_exog

    def create_dictionaries(self, series_df: pd.DataFrame, exog_df: pd.DataFrame, future_exog_df: pd.DataFrame) -> tuple[series_long_to_dict, exog_long_to_dict, exog_long_to_dict]:
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

    def train_cat_forecaster(self, series_dict: dict, exog_dict: exog_long_to_dict) -> ForecasterAutoregMultiSeries:
        regressor = CatBoostRegressor(random_state=123, max_depth=5, silent=True)
        # regressor = LGBMRegressor(random_state=123, max_depth=5)
        print(regressor.get_params())
        forecaster = ForecasterAutoregMultiSeries(
                        regressor          = regressor,
                        lags               = 10,
                        encoding           = "ordinal",
                        dropna_from_series = False
                    )
        
        forecaster.fit(series=series_dict, exog=exog_dict,suppress_warnings=True)
        
        return forecaster

    def train_lgbm_forecaster(self, series_dict: dict, exog_dict: exog_long_to_dict) -> ForecasterAutoregMultiSeries:
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

    def predict_X_days(self, days_to_predict: int, forecaster: ForecasterAutoregMultiSeries, future_exog_dict: exog_long_to_dict) -> pd.DataFrame:
        return forecaster.predict(steps=days_to_predict, exog=future_exog_dict, suppress_warnings=True)

    def inverse_scaling(self, train_df: pd.DataFrame, pred_df: pd.DataFrame, scalers: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = train_df.copy()
        pred = pred_df.copy()

        # Desescalar los datos de entrenamiento y prueba por token
        for token_id, scaler in scalers.items():
            # Aplicar el inverso del escalado en 'close' para cada token
            if token_id in train['id'].unique():
                train.loc[train['id'] == token_id, 'close'] = scaler.inverse_transform(train[train['id'] == token_id][['close']])

            # Desescalar las predicciones para cada columna del token en pred_df
            if token_id in pred.columns:
                pred[token_id] = scaler.inverse_transform(pred[[token_id]])

        return train, pred

    def preprocess(self, data: pd.DataFrame, days_to_predict: int = 7):
        data_cleaned = self.removing_duplicates(data)
        data_capped = self.capping_time_series(data_cleaned)
        data_datetime = self.timestamp_to_datetime(data_capped)
        data_datetime.reset_index(drop=True, inplace=True)
        data_final, series_scaler, exog_scaler = self.scaling(data_datetime)
        series, exog = self.create_series_exog(data_final)
        future_exog = self.create_all_future_exog(data_final, exog_scaler=exog_scaler, days=days_to_predict)
        series_dict, exog_dict, future_exog_dict = self.create_dictionaries(series, exog, future_exog)

        self.train_data = data_final
        self.series_dict = series_dict
        self.exog_dict = exog_dict
        self.future_exog_dict = future_exog_dict
        self.series_scaler = series_scaler
        self.exog_scaler = exog_scaler
    
    def predict(self, model: str = 'catboost'):
        if model == 'catboost':
            self.forecaster = self.train_cat_forecaster(self.series_dict, self.exog_dict)
        elif model == 'lgbm':
            self.forecaster = self.train_lgbm_forecaster(self.series_dict, self.exog_dict)
        else:
            raise ValueError('Model not supported')

        self.predictions = self.predict_X_days(days_to_predict=self.days_to_predict, forecaster=self.forecaster, future_exog_dict=self.future_exog_dict)
        self.og_train, self.og_pred = self.inverse_scaling(train_df=self.train_data, pred_df=self.predictions, scalers=self.series_scaler)
        self.complete_time_series = self.generate_complete_time_series(hist_df=self.og_train, pred_df=self.og_pred, category=self.og_train['category'].iloc[0])

    def generate_complete_time_series(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, category: str):
        ids = list(pred_df.columns)

        pred = pred_df.copy()
        hist = hist_df.copy()

        close_df = hist.groupby('id').tail(1)[['id', 'close']]

        for i in ids:
            close_df.loc[close_df['id'] == i, 'pred_close'] = pred[i].tail(1).values[0]
            close_df['increase'] = (close_df['pred_close'] > close_df['close']).astype(int)
        close_df.reset_index(drop=True, inplace=True)

        close_df.columns = ['id', 'last_close', 'last_pred_close', 'has_increased']

        hist = hist[['timestamp', 'id', 'name', 'symbol', 'category', 'close']]
        pred = pred.stack().reset_index()
        pred.columns = ['timestamp', 'id', 'close']

        metadata_df = hist[['id', 'name', 'symbol', 'category']].drop_duplicates()

        pred = pred.merge(metadata_df, on='id', how='left')

        final_df = pd.concat([hist, pred], ignore_index=True)
        final_df = final_df.sort_values(['id', 'timestamp']).reset_index(drop=True)

        for i in ids:
            final_df.loc[final_df['id'] == i, ['last_close', 'last_pred_close', 'has_increased']] = close_df.loc[close_df['id'] == i, ['last_close', 'last_pred_close', 'has_increased']].values

        final_df['timestamp'] = final_df['timestamp'].astype(str)

        result = (
            final_df.groupby(["id", "name", "symbol", "category", 'last_close', 'last_pred_close', 'has_increased'])
            .apply(lambda group: group[["timestamp", "close"]].to_dict(orient="records"))
            .reset_index(name="close_data")
            .to_dict(orient="records")
        )

        json_result = json.dumps(result, indent=4)

        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')
        # save
        with open(f'../data/processed/{category}_complete_time_series.json', 'w') as f:
            f.write(json_result)

        return json_result