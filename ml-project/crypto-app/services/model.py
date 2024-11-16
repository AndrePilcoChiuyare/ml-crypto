import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict
from datetime import timedelta, datetime
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from catboost import CatBoostRegressor
import json
import os
import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

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
        
    def save_json(self, json_variable: list, category: str):
        with open(f'../data/processed/{category}.json', 'w') as json_file:
                json.dump(json_variable, json_file, indent=4)

    def get_last_date(self, category: str):
        if category == "ai":
            token_id = "040f0133-1654-4e4e-85ac-417155ca814f"
        elif category == "gaming":
            token_id = "def04c24-d3a3-4b80-8eb1-98f9a91c80c9"
        elif category == "meme":
            token_id = "02e9c2cc-2e3b-45fe-b7bb-508cb23a3a39"
        elif category == "rwa":
            token_id = "1c1cd416-b027-4d73-9d4d-0a9edc63524d"
        data = self.loadJSON(f'../data/processed/{category}_complete_time_series.json')
        last_date = data[token_id]['close_data'][-8]['timestamp']
        return last_date
        
    def get_basic_prediction_info(self, category:str):
        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')
        tokens = self.loadJSON(f'../data/processed/{category}_complete_time_series.json')

        for token in tokens.values():
            # remove the key "close_data" from the dictionary
            token.pop('close_data', None)
        
        return tokens
    
    def get_prediction_by_id(self, category:str, token_id: str):
        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')
        tokens = self.loadJSON(f'../data/processed/{category}_complete_time_series.json')

        return tokens[token_id]

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
        # mean = np.floor(token_info.mean()).astype(int)
        # ids_to_keep = token_info[token_info > mean].index.get_level_values(0).unique()
        # filtered_meme = df[df['id'].isin(ids_to_keep)]
        filtered_meme = df.copy()
        first_timestamps = filtered_meme.groupby('id').timestamp.min()
        last_timestamps = filtered_meme.groupby('id').timestamp.max()
        max_first_timestamp = first_timestamps.max()
        last_timestamp = last_timestamps.max()
        print(filtered_meme['category'].iloc[0], last_timestamp)
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
        self.of_pred = self.reemplazar_valores_negativos_ponderada(self.og_pred, self.og_train)
        self.complete_time_series = self.generate_complete_time_series(hist_df=self.og_train, pred_df=self.og_pred, category=self.og_train['category'].iloc[0])

    def market_cap_level(self, row):
        if row['category'] == 'meme':
            return 'low' if row['marketcap'] < 100e6 else 'high'
        elif row['category'] == 'ai':
            return 'low' if row['marketcap'] < 30e6 else 'high'
        elif row['category'] == 'rwa':
            return 'low' if row['marketcap'] < 150e6 else 'high'
        elif row['category'] == 'gaming':
            return 'low' if row['marketcap'] < 50e6 else 'high'
        return 'unknown'

    def generate_complete_time_series(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, category: str):
        ids = list(pred_df.columns)

        pred = pred_df.copy()
        hist = hist_df.copy()

        close_df = hist.groupby('id').tail(1)[['id', 'close']]

        for i in ids:
            close_df.loc[close_df['id'] == i, 'pred_close'] = pred[i].tail(1).values[0]
            close_df['increase'] = (close_df['pred_close'] > close_df['close']).astype(int)
            close_df['future_multiply'] = round(close_df['pred_close'] / close_df['close'], 2)
        close_df.reset_index(drop=True, inplace=True)

        close_df.columns = ['id', 'last_close', 'last_pred_close', 'has_increased', 'future_multiply']

        hist = hist[['timestamp', 'id', 'name', 'symbol', 'category', 'marketcap', 'image','close']]
        hist['market_cap_level'] = hist.apply(self.market_cap_level, axis=1)
        pred = pred.stack().reset_index()
        pred.columns = ['timestamp', 'id', 'close']

        metadata_df = hist[['id', 'name', 'symbol', 'category', 'marketcap', 'market_cap_level', 'image']].drop_duplicates()

        pred = pred.merge(metadata_df, on='id', how='left')

        final_df = pd.concat([hist, pred], ignore_index=True)
        final_df = final_df.sort_values(['id', 'timestamp']).reset_index(drop=True)

        for i in ids:
            final_df.loc[final_df['id'] == i, ['last_close', 'last_pred_close', 'has_increased', 'future_multiply']] = close_df.loc[close_df['id'] == i, ['last_close', 'last_pred_close', 'has_increased', 'future_multiply']].values

        final_df['timestamp'] = final_df['timestamp'].astype(str)
        final_df['token_id'] = final_df['id']

        result = (
            final_df.groupby(["id", "token_id", "name", "symbol", "category", 'last_close', 'last_pred_close', 'has_increased', 'future_multiply', 'marketcap', 'market_cap_level', 'image'])
            .apply(lambda group: group[["timestamp", "close"]].to_dict(orient="records"))
            .reset_index(name="close_data")
            .set_index("token_id")
            .to_dict(orient="index")
        )              

        if category not in ["meme", "ai", "rwa", "gaming"]:
            raise ValueError('Category not listed')

        json_result = json.dumps(result, indent=4)   
        
        # save
        with open(f'../data/processed/{category}_complete_time_series.json', 'w') as f:
            f.write(json_result)

        return json_result
    
    def media_movil_ponderada(self, valores):
        n = len(valores)
        pesos = range(1, n + 1)
        return sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)
    
    def obtener_valores_anteriores(self, df, columna, indice_negativo, cantidad=14):
        inicio = max(indice_negativo - cantidad, 0)
        return df[columna].iloc[inicio:indice_negativo + 1].tolist()
    
    def completar_valores_con_train(self, df_pred, df_train, id_moneda, fecha_negativa, cantidad_total=15):
        # Filtrar los datos de entrenamiento por el ID de la moneda
        datos_moneda = df_train[df_train['id'] == id_moneda]
        valores_close = []

        # Obtener el valor de close en la fecha negativa
        valor_fecha_negativa = datos_moneda[datos_moneda['timestamp'] == fecha_negativa]['close']

        if not valor_fecha_negativa.empty:
            valores_close.append(valor_fecha_negativa.values[0])

        # Obtener valores de close hasta completar 15 valores
        for i in range(1, cantidad_total):
            fecha_anterior = pd.to_datetime(fecha_negativa) - pd.Timedelta(days=i)
            valor_anterior = datos_moneda[datos_moneda['timestamp'] == fecha_anterior.strftime('%Y-%m-%d')]
            
            if not valor_anterior.empty:
                valores_close.append(valor_anterior['close'].values[0])
            else:
                break

        return valores_close
    
    def reemplazar_valores_negativos_ponderada(self, df_pred, df_train):
        for columna in df_pred.columns:
            for i in range(len(df_pred)):
                valor_actual = df_pred[columna].iloc[i]

                if valor_actual < 0:
                    # Obtener 14 valores anteriores
                    valores_anteriores = self.obtener_valores_anteriores(df_pred, columna, i)

                    # Si hay suficientes valores, calcular la media
                    if len(valores_anteriores) == 15:
                        media_ponderada = self.media_movil_ponderada(valores_anteriores)
                    else:
                        # Completar con datos de gaming_obj.og_train
                        fecha_negativa = df_pred.index[i]
                        id_moneda = columna
                        valores_necesarios = self.completar_valores_con_train(df_pred, df_train, id_moneda, fecha_negativa)
                        # Concatenar valores anteriores con los obtenidos de og_train
                        valores_totales = valores_anteriores + valores_necesarios
                        media_ponderada = self.media_movil_ponderada(valores_totales)

                    # Reemplazar el valor negativo con la media móvil ponderada
                    df_pred[columna].iloc[i] = max(media_ponderada, 0)

        return df_pred
    
    def add_cols_to_df(self, df, df_marketcap, df_image):
        df_csv = df.copy()
        
        df_csv['marketcap'] = int(-1)
        df_csv['image'] = "https://cdn-icons-png.flaticon.com/512/272/272525.png"

        for _, row in df_marketcap.iterrows():
            df_csv.loc[df_csv['id'] == row['id'], 'marketcap'] = row['marketcap']

        for _, row in df_image.iterrows():
            df_csv.loc[df_csv['id'] == row['id'], 'image'] = row['image']
        
        return df_csv
    
    def getData(self):
        meme = self.loadJSON("../data/processed/meme-pre-filtered.json")
        ai = self.loadJSON("../data/processed/ai-pre-filtered.json")
        rwa = self.loadJSON("../data/processed/rwa-pre-filtered.json")
        gaming = self.loadJSON("../data/processed/gaming-pre-filtered.json")

        print("meme start")
        self.get_yearly_data('30/09/2024', '1d', meme, 'meme')
        print("ai start")
        self.get_yearly_data('30/09/2024', '1d', ai, 'ai')
        print("rwa start")
        self.get_yearly_data('30/09/2024', '1d', rwa, 'rwa')
        print("gaming start")
        self.get_yearly_data('30/09/2024', '1d', gaming, 'gaming')

        print("data complete")
        meme_complete = self.complete_historical_data('meme')
        ai_complete = self.complete_historical_data('ai')
        rwa_complete = self.complete_historical_data('rwa')
        gaming_complete = self.complete_historical_data('gaming')

        print("saving complete json")
        self.save_json(meme_complete, 'meme')
        self.save_json(ai_complete, 'ai')
        self.save_json(rwa_complete, 'rwa')
        self.save_json(gaming_complete, 'gaming')

        print(type(meme_complete))

        print("complete to df")
        meme_data = self.historical_json_to_dataframe(meme_complete)
        ai_data = self.historical_json_to_dataframe(ai_complete)
        rwa_data = self.historical_json_to_dataframe(rwa_complete)
        gaming_data = self.historical_json_to_dataframe(gaming_complete)

        print("saving data csv")
        meme_data.to_csv("../data/processed/meme.csv", index=False)
        ai_data.to_csv("../data/processed/ai.csv", index=False)
        rwa_data.to_csv("../data/processed/rwa.csv", index=False)
        gaming_data.to_csv("../data/processed/gaming.csv", index=False)

        # gaming_data = pd.read_csv("../data/processed/gaming.csv")
        # meme_data = pd.read_csv("../data/processed/meme.csv")
        # ai_data = pd.read_csv("../data/processed/ai.csv")
        # rwa_data = pd.read_csv("../data/processed/rwa.csv")
        # print("data loaded")

        gaming_cap = pd.read_csv("../data/processed/gaming_market_caps.csv")
        meme_cap = pd.read_csv("../data/processed/meme_market_caps.csv")
        ai_cap = pd.read_csv("../data/processed/ai_market_caps.csv")
        rwa_cap = pd.read_csv("../data/processed/rwa_market_caps.csv")
        print("market caps loaded")

        gaming_image = pd.read_csv("../data/processed/gaming_logos.csv")
        meme_image = pd.read_csv("../data/processed/meme_logos.csv")
        ai_image = pd.read_csv("../data/processed/ai_logos.csv")
        rwa_image = pd.read_csv("../data/processed/rwa_logos.csv")
        print("images loaded")

        gaming_data = self.add_cols_to_df(gaming_data, gaming_cap, gaming_image)
        meme_data = self.add_cols_to_df(meme_data, meme_cap, meme_image)
        ai_data = self.add_cols_to_df(ai_data, ai_cap, ai_image)
        rwa_data = self.add_cols_to_df(rwa_data, rwa_cap, rwa_image)
        print("cols added")

        gaming_data.to_csv("../data/processed/gaming.csv", index=False)
        meme_data.to_csv("../data/processed/meme.csv", index=False)
        ai_data.to_csv("../data/processed/ai.csv", index=False)
        rwa_data.to_csv("../data/processed/rwa.csv", index=False)
        print("data saved")

    def get_from_api(self, endpoint: str):
        base_url = 'https://api.messari.io/'
        url = f'{base_url}{endpoint}'
        headers = {
            'accept': 'application/json',
            'x-messari-api-key': os.getenv('MESSARI_ANDRE_KEY'),
        }
        return requests.get(url, headers=headers)
    
    def getting_interval_timestamps(self, date: str, days:int) -> tuple[int, int]:
        startdate: datetime = datetime.strptime(date, '%d/%m/%Y')
        enddate: datetime = startdate + timedelta(days=days)
        return int(datetime.timestamp(startdate)), int(datetime.timestamp(enddate))
    
    def get_yearly_data(self, date_since:str, interval:str, tokens: list, category: str)->None:
        year = date_since

        while(1):
            print(f"Getting data since {year}")
            annual_data = []

            days = 360

            if datetime.now() - timedelta(days=360) < datetime.strptime(year, "%d/%m/%Y"):
                days = int((datetime.now() - datetime.strptime(year, "%d/%m/%Y")).days)
            
            start_timestamp, end_timestamp = self.getting_interval_timestamps(year, days)
            print(f"{start_timestamp} - {end_timestamp} ({days} days)")
            
            for token in tokens:
                new_token = token.copy()
                del new_token['allTimeHighData']
                del new_token['cycleLowData']
                new_token['category'] = category
                endpoint = f"marketdata/v1/assets/{token['id']}/price/time-series?interval={interval}&startTime={start_timestamp}&endTime={end_timestamp}"
                response = self.get_from_api(endpoint)
                if response.status_code == 200:
                    result = response.json()
                    new_token['market_data'] = result['data']
                else:
                    new_token['market_data'] = "No content"
                annual_data.append(new_token)
            
            with open(f'../data/raw/{category}/{category}-{year[6:]}.json', 'w') as json_file:
                json.dump(annual_data, json_file, indent=4)
            
            # updating year
            next_year = datetime.strptime(year, "%d/%m/%Y") + timedelta(days=days)
            year = next_year.strftime("%d/%m/%Y")

            if days < 360:
                break
    
    def complete_historical_data(self, category: str):
        complete_data = []
        for file in os.listdir(f'../data/raw/{category}'):
            with open(f'../data/raw/{category}/{file}') as json_file:
                data = json.load(json_file)
                if complete_data == []:
                    complete_data = data
                else:
                    for record in complete_data:
                        for new_record in data:
                            if record['id'] == new_record['id']:
                                if record['market_data'] == "No content":
                                    record['market_data'] = new_record['market_data']
                                elif new_record['market_data'] != "No content":
                                    record['market_data'] += new_record['market_data']
                                break
                                    
        return complete_data
    
    def historical_json_to_dataframe(self, json_file: list) -> pd.DataFrame:
        data = []
        for token in json_file:
            for record in token['market_data']:
                new_record = record.copy()
                new_record['name'] = token['name']
                new_record['symbol'] = token['symbol']
                new_record['id'] = token['id']
                new_record['category'] = token['category']
                data.append(new_record)
        return pd.DataFrame(data)
