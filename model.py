from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def create_engineered_features(df):
    """Create optimized feature set based on correlation analysis"""
    df = df.copy()

    #Import holidays data
    holidays = pd.read_csv('external_data/Holidays/Propre_jours_feries.csv')
    holidays_sco = pd.read_csv('external_data/Holidays/Propre_vacances_scolaires.csv')

    #Define holidays columns and holidays_sco
    holidays['date'] = pd.to_datetime(holidays['date'])
    df['is_holiday'] = df['date'].dt.date.isin(holidays['date'].dt.date)
    holidays_sco['date'] = pd.to_datetime(holidays_sco['date'])
    df['is_holiday_sco'] = df['date'].dt.date.isin(holidays_sco['date'].dt.date)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.weekday >= 5

    # Keep only hour cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Create working day feature
    df['is_working_day'] = (~df['is_weekend']) & (~df['is_holiday'])
    # Create rush hour feature (only for working days)
    morning_rush = (df['hour'].between(7, 9)) & df['is_working_day']
    evening_rush = (df['hour'].between(16, 18)) & df['is_working_day']
    # Add rush hour binary feature
    df['is_rush_hour'] = (morning_rush | evening_rush).astype(bool)

    # Drop useless columns
    df = df.drop(columns='hour')
    df = df.drop(columns='month')
    df = df.drop(columns=['counter_id', 'site_id', 'site_name', 'bike_count',
       'date', 'counter_installation_date', 'coordinates',
       'counter_technical_id'])


    # Define feature groups
    numeric_features = [
        'latitude',
        'longitude',
        'hour_sin',
        'hour_cos',
        'month_sin',
        'month_cos'
    ]

    binary_features = [
        'is_weekend',
        'is_holiday',
        'is_holiday_sco',
        'is_working_day',
        'is_rush_hour'
    ]

    categorical_features = ['counter_name', 'weekday']

    target = ['log_bike_count']

    reordered_columns = numeric_features + binary_features + categorical_features + target
    df = df.reindex(columns=reordered_columns)


    return df, numeric_features, binary_features, categorical_features

def prepare_data(df):
    # Apply feature engineering
    transformed_df, num_feat, bin_feat, cat_feat = create_engineered_features(df)

    # Split features and target
    X = transformed_df.drop(columns=['log_bike_count'])
    y = transformed_df['log_bike_count']

    return X, y, num_feat, bin_feat, cat_feat

def create_pipeline(self):
        # Add polynomial features for numeric variables
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features),
            ('bin', 'passthrough', self.binary_features)
        ])

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge())
        ])