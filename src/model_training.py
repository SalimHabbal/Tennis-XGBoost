import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from src.data_processing import load_data, calculate_elo, compute_rolling_stats, calculated_h2h, prepare_features
import os

def get_processed_dataset(data_dir):
    """
    Runs the full data processing pipeline and returns the features dataframe.
    """
    print("Loading and processing data...")
    df = load_data(data_dir)
    df = calculate_elo(df)
    df = compute_rolling_stats(df)
    df = calculated_h2h(df)
    final_df = prepare_features(df)
    return final_df

def train_and_eval(data_dir):
    """
    Trains the XGBoost model and evaluates it.
    """
    df = get_processed_dataset(data_dir)
    
    # Define features and target
    features = [
        'ace_diff', 'df_diff', 'bp_diff', 'h2h_diff', 
        'elo_diff', 'p1_elo', 'p2_elo', 'surface'
    ]
    target = 'label'
    
    # Preprocessing for categorical 'surface'
    # XGBoost supports categorical data with enable_categorical=True, 
    # but the columns must be 'category' dtype.
    df['surface'] = df['surface'].astype('category')
    
    # Chronological Split
    # Train: 2015-2023
    # Test: 2024
    
    df['year'] = df['tourney_date'].dt.year
    train_df = df[df['year'] < 2024]
    test_df = df[df['year'] == 2024]
    
    print(f"Train matches: {len(train_df)}")
    print(f"Test matches: {len(test_df)}")
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    # XGBoost Classifier
    # Using some reasonable default hyperparameters to control overfitting
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        enable_categorical=True,
        early_stopping_rounds=50,
        eval_metric='logloss',
        random_state=42
    )
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    
    print("\n" + "="*30)
    print(f"Results for Test Year 2024:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("="*30)
    
    # Feature Importance
    print("\nFeature Importances:")
    importances = model.feature_importances_
    for name, imp in zip(features, importances):
        print(f"{name}: {imp:.4f}")
        
    return model, acc

if __name__ == "__main__":
    # For quick testing
    import sys
    data_path = "/Users/salimhabbal/Desktop/Tennis XGBoost/data/raw"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    train_and_eval(data_path)
