from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from model_catboost_meteo_covid import prepare_data
from optuna_tuning_distance import add_simple_distance_feature

if __name__ == "__main__":
    # Load data
    train_df = pd.read_parquet(Path("data/train.parquet"))
    test_df = pd.read_parquet(Path("data/final_test.parquet"))
    weather_df = pd.read_csv('external_data/Weather/Propre_meteo.csv')
    covid_df = pd.read_csv('external_data/Covid/Propre_nbr_Covid.csv')

    # Add distance feature to both train and test
    train_df = add_simple_distance_feature(train_df)
    test_df = add_simple_distance_feature(test_df)

    # Prepare features
    X_train, y_train = prepare_data(train_df, weather_df, covid_df, is_train=True)
    X_test = prepare_data(test_df, weather_df, covid_df, is_train=False)

    # Trial 12 parameters (better ones)
    parameters = {
        'iterations': 489,
        'learning_rate': 0.262,
        'depth': 9,
        'bagging_temperature': 0.432,
        'border_count': 109,
        'od_wait': 27
    }

    # Train ensemble
    predictions_list = []
    for seed in [42, 123, 456]:
        model = CatBoostRegressor(
            **parameters,  # using parameters instead of params
            random_seed=seed,
            cat_features=['counter_name', 'weekday']
        )
        model.fit(X_train, y_train, verbose=False)
        predictions_list.append(model.predict(X_test))

    # Average predictions
    predictions = np.mean(predictions_list, axis=0)

    # Save predictions
    pd.DataFrame({
        'Id': range(len(predictions)),
        'log_bike_count': predictions
    }).to_csv('submission_with_distance.csv', index=False)