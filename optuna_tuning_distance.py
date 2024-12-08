import optuna
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from model_catboost_meteo_covid import prepare_data

def add_simple_distance_feature(df):
    """Add simple binary distance feature based on median Euclidean distance to nearest station"""
    df = df.copy()

    stations = {}
    for _, row in df.drop_duplicates('counter_name')[['counter_name', 'latitude', 'longitude']].iterrows():
        stations[row['counter_name']] = (row['latitude'], row['longitude'])

    nearest_dist = {}
    for s1, coords1 in stations.items():
        distances = []
        for s2, coords2 in stations.items():
            if s1 != s2:
                dist = ((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)**0.5
                distances.append(dist)
        nearest_dist[s1] = min(distances)

    median_dist = np.median(list(nearest_dist.values()))
    df['is_isolated'] = df['counter_name'].map(lambda x: 1 if nearest_dist[x] > median_dist else 0)

    return df

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 450, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.2, 0.28),
        'depth': trial.suggest_int('depth', 7, 9),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.4, 0.5),
        'border_count': trial.suggest_int('border_count', 100, 130),
        'od_type': 'IncToDec',
        'od_wait': trial.suggest_int('od_wait', 20, 30),
    }

    scores = []
    # Using TimeSeriesSplit with 3 splits
    tscv = TimeSeriesSplit(n_splits=3, test_size=24*7)  # 1 week test size

    for train_idx, val_idx in tscv.split(X_sample):
        X_fold_train, X_fold_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_fold_train, y_fold_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        model = CatBoostRegressor(
            **params,
            random_seed=42,
            cat_features=['counter_name', 'weekday']
        )

        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=50,
            verbose=False
        )

        pred = model.predict(X_fold_val)
        rmse = np.sqrt(np.mean((y_fold_val - pred) ** 2))
        scores.append(rmse)

    return np.mean(scores)

if __name__ == "__main__":
    # Load training data
    train_df = pd.read_parquet(Path("data/train.parquet"))
    weather_df = pd.read_csv('external_data/Weather/Propre_meteo.csv')
    covid_df = pd.read_csv('external_data/Covid/Propre_nbr_Covid.csv')

    # Add distance feature
    train_df = add_simple_distance_feature(train_df)

    # Prepare features
    X_train, y_train = prepare_data(train_df, weather_df, covid_df, is_train=True)

    # Sample data while maintaining temporal order
    # Taking every Nth row to maintain temporal patterns
    N = 4  # adjust this to control sample size
    X_sample = X_train.iloc[::N].copy()
    y_sample = y_train.iloc[::N].copy()

    print(f"Original data size: {len(X_train)}")
    print(f"Sample data size: {len(X_sample)}")

    # Run optimization study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)  # Reduced number of trials

    print("\nBest parameters:", study.best_params)
    print("Best RMSE:", study.best_value)

    # Train final model on full dataset with best parameters to check feature importance
    final_model = CatBoostRegressor(
        **study.best_params,
        random_seed=42,
        cat_features=['counter_name', 'weekday']
    )
    final_model.fit(X_train, y_train, verbose=False)

    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 feature importances:")
    print(importances.head(10))