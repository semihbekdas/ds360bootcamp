import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Veriyi preprocessing işleminden geçirir
        """
        df_processed = df.copy()
        
        # Target column'u bul
        target_candidates = ['default', 'loan_status', 'target', 'class', 'y']
        target_col = None
        for col in df_processed.columns:
            if any(candidate in col.lower() for candidate in target_candidates):
                target_col = col
                break
        
        if target_col is None:
            # Eğer target bulunamazsa binary olan ilk kolonu al
            for col in df_processed.columns:
                if df_processed[col].nunique() == 2:
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Target column bulunamadı!")
        
        print(f"Target column: {target_col}")
        
        # Features ve target'ı ayır
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Kategorik değişkenleri encode et
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Feature isimlerini sakla
        self.feature_names = X.columns.tolist()
        
        # Target'ı binary yap
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.target_encoder = le_target
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
    def get_balanced_datasets(self, X, y, test_size=0.2, random_state=42):
        """
        SMOTE, undersampling ve class weights için veri setleri hazırlar
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Class weights hesapla
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # SMOTE uygula
        smote = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Random undersampling uygula
        undersampler = RandomUnderSampler(random_state=random_state)
        X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
        
        datasets = {
            'original': (X_train, X_test, y_train, y_test),
            'smote': (X_train_smote, X_test, y_train_smote, y_test),
            'undersampled': (X_train_under, X_test, y_train_under, y_test),
            'class_weights': class_weight_dict
        }
        
        # Dataset istatistikleri yazdır
        print("Dataset İstatistikleri:")
        print(f"Original - Train: {len(y_train)}, Positive: {sum(y_train)}, Negative: {len(y_train)-sum(y_train)}")
        print(f"SMOTE - Train: {len(y_train_smote)}, Positive: {sum(y_train_smote)}, Negative: {len(y_train_smote)-sum(y_train_smote)}")
        print(f"Undersampled - Train: {len(y_train_under)}, Positive: {sum(y_train_under)}, Negative: {len(y_train_under)-sum(y_train_under)}")
        print(f"Test: {len(y_test)}, Positive: {sum(y_test)}, Negative: {len(y_test)-sum(y_test)}")
        
        return datasets

def load_and_preprocess():
    """
    Veriyi yükleyip preprocessing yapar
    """
    from data_loader import load_data
    
    # Veriyi yükle
    df = load_data()
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(df)
    
    # Balanced datasets oluştur
    datasets = preprocessor.get_balanced_datasets(X, y)
    
    return datasets, preprocessor

if __name__ == "__main__":
    datasets, preprocessor = load_and_preprocess()
    print("Preprocessing tamamlandı!")