import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, 
                                method='original', class_weight=None):
        """
        Logistic Regression modelini eğitir
        """
        print(f"\nLogistic Regression eğitiliyor - {method} dataset...")
        
        if class_weight is not None:
            lr = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000)
        else:
            lr = LogisticRegression(random_state=42, max_iter=1000)
        
        lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        model_name = f'lr_{method}'
        self.models[model_name] = lr
        self.results[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc_score': auc_score,
            'method': method
        }
        
        print(f"AUC Score: {auc_score:.4f}")
        return lr
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, 
                     method='original', class_weight=None):
        """
        XGBoost modelini eğitir
        """
        print(f"\nXGBoost eğitiliyor - {method} dataset...")
        
        if class_weight is not None:
            # XGBoost için scale_pos_weight hesapla
            scale_pos_weight = class_weight[0] / class_weight[1]
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
        else:
            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        model_name = f'xgb_{method}'
        self.models[model_name] = xgb_model
        self.results[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc_score': auc_score,
            'method': method
        }
        
        print(f"AUC Score: {auc_score:.4f}")
        return xgb_model
    
    def train_all_models(self, datasets):
        """
        Tüm modelleri farklı veri setleri ile eğitir
        """
        print("Tüm modeller eğitiliyor...")
        
        # Original dataset
        X_train, X_test, y_train, y_test = datasets['original']
        self.train_logistic_regression(X_train, y_train, X_test, y_test, 'original')
        self.train_xgboost(X_train, y_train, X_test, y_test, 'original')
        
        # Class weights ile
        class_weights = datasets['class_weights']
        self.train_logistic_regression(X_train, y_train, X_test, y_test, 'class_weights', class_weights)
        self.train_xgboost(X_train, y_train, X_test, y_test, 'class_weights', class_weights)
        
        # SMOTE dataset
        X_train_smote, _, y_train_smote, _ = datasets['smote']
        self.train_logistic_regression(X_train_smote, y_train_smote, X_test, y_test, 'smote')
        self.train_xgboost(X_train_smote, y_train_smote, X_test, y_test, 'smote')
        
        # Undersampled dataset
        X_train_under, _, y_train_under, _ = datasets['undersampled']
        self.train_logistic_regression(X_train_under, y_train_under, X_test, y_test, 'undersampled')
        self.train_xgboost(X_train_under, y_train_under, X_test, y_test, 'undersampled')
        
        print("\nTüm modeller eğitildi!")
    
    def evaluate_models(self):
        """
        Tüm modelleri değerlendirir
        """
        print("\nModel Değerlendirme Sonuçları:")
        print("="*50)
        
        results_df = []
        for model_name, result in self.results.items():
            auc = result['auc_score']
            method = result['method']
            model_type = 'Logistic Regression' if 'lr_' in model_name else 'XGBoost'
            
            results_df.append({
                'Model': model_type,
                'Method': method,
                'AUC Score': auc
            })
            
            print(f"{model_type} ({method}): AUC = {auc:.4f}")
        
        results_df = pd.DataFrame(results_df)
        return results_df
    
    def plot_roc_curves(self):
        """
        ROC curve'leri çizer
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
            auc = result['auc_score']
            method = result['method']
            model_type = 'LR' if 'lr_' in model_name else 'XGB'
            
            plt.plot(fpr, tpr, label=f'{model_type} ({method}) - AUC: {auc:.3f}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Karşılaştırması')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_best_model(self):
        """
        En iyi modeli döndürür
        """
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name]['auc_score']
        
        print(f"En iyi model: {best_model_name} (AUC: {best_score:.4f})")
        return best_model, best_model_name
    
    def save_models(self, save_dir='../models'):
        """
        Modelleri kaydeder
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            file_path = os.path.join(save_dir, f'{model_name}.pkl')
            joblib.dump(model, file_path)
            print(f"{model_name} kaydedildi: {file_path}")

def train_models():
    """
    Ana eğitim fonksiyonu
    """
    from preprocessing import load_and_preprocess
    
    # Veriyi yükle ve preprocess et
    datasets, preprocessor = load_and_preprocess()
    
    # Model eğitimcisini oluştur
    trainer = ModelTrainer()
    
    # Tüm modelleri eğit
    trainer.train_all_models(datasets)
    
    # Sonuçları değerlendir
    results_df = trainer.evaluate_models()
    
    # ROC curve'leri çiz
    trainer.plot_roc_curves()
    
    # En iyi modeli bul
    best_model, best_model_name = trainer.get_best_model()
    
    # Modelleri kaydet
    trainer.save_models()
    
    return trainer, preprocessor, best_model_name

if __name__ == "__main__":
    trainer, preprocessor, best_model_name = train_models()
    print(f"\nEğitim tamamlandı! En iyi model: {best_model_name}")