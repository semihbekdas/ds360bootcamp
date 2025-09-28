import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

def train_model(model_type="random_forest", test_size=0.2, random_state=42):
    data = pd.read_csv('data/processed/iris_processed.csv')
    
    

    X = data.drop('species', axis=1)
    y = data['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y 

    )

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:  
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path)
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('models/features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f" Model trained: {model_type}")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Model saved: {model_path}")
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = train_model(model_type='random_forest')

    model_lr, metrics_lr = train_model(model_type='logistic_regression')

    print("\nBoth models have been trained and saved!")