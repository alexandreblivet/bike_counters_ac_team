from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def prepare_data(df, is_train=None):
    """Create optimized feature set based on correlation analysis"""
    df = df.copy()

    # Import holidays data
    holidays = pd.read_csv('external_data/Holidays/Propre_jours_feries.csv')
    holidays_sco = pd.read_csv('external_data/Holidays/Propre_vacances_scolaires.csv')

    # Add holiday features
    holidays['date'] = pd.to_datetime(holidays['date'])
    df['is_holiday'] = df['date'].dt.date.isin(holidays['date'].dt.date)
    holidays_sco['date'] = pd.to_datetime(holidays_sco['date'])
    df['is_holiday_sco'] = df['date'].dt.date.isin(holidays_sco['date'].dt.date)

    # Time features
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.weekday >= 5

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Rush hour
    df['is_working_day'] = (~df['is_weekend']) & (~df['is_holiday'])
    morning_rush = (df['hour'].between(7,9)) & df['is_working_day']
    evening_rush = (df['hour'].between(16,18)) & df['is_working_day']
    df['is_rush_hour'] = (morning_rush | evening_rush)

    # Drop unused columns
    cols_to_drop = ['counter_id', 'site_id', 'site_name', 'counter_installation_date',
                    'coordinates', 'counter_technical_id', 'hour', 'month', 'date']
    if 'bike_count' in df.columns:
        cols_to_drop.append('bike_count')

    df = df.drop(columns=cols_to_drop)

    if is_train:
        X = df.drop(columns=['log_bike_count'])
        y = df['log_bike_count']
        return X, y
    return df

if __name__ == "__main__":
    train_df = pd.read_parquet(Path("data/train.parquet"))
    test_df = pd.read_parquet(Path("data/final_test.parquet"))

    X_train, y_train = prepare_data(train_df, is_train=True)
    X_test = prepare_data(test_df, is_train=False)

    model = CatBoostRegressor(
        iterations=469,
        learning_rate=0.236,
        depth=8,
        bagging_temperature=0.45,
        border_count=115,
        od_type='IncToDec',
        od_wait=26,
        random_seed=42,
        cat_features=['counter_name', 'weekday']
    )

    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)

    pd.DataFrame({
        'Id': range(len(predictions)),
        'log_bike_count': predictions
    }).to_csv('submission.csv', index=False)