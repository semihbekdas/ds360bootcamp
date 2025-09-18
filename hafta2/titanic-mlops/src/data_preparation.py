import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_titanic_data():
    """Basit Titanic veri seti oluştur"""
    
    # Basit Titanic veri seti (gerçek veri yerine)
    np.random.seed(42)
    n_samples = 800
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'Age': np.random.normal(30, 12, n_samples),
        'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'Parch': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        'Fare': np.random.exponential(30, n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Survived hedefini mantıklı kurallara göre oluştur
    survived_prob = np.where(
        (df['Sex'] == 'female') & (df['Pclass'] <= 2), 0.8,
        np.where(df['Sex'] == 'female', 0.6,
        np.where(df['Pclass'] == 1, 0.4, 0.2))
    )
    
    df['Survived'] = np.random.binomial(1, survived_prob)
    
    # Bazı Age değerlerini NaN yap
    missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[missing_indices, 'Age'] = np.nan
    
    return df

def prepare_data(df):
    """Veriyi model için hazırla"""
    
    # Kopyasını al
    df = df.copy()
    
    # Age eksik değerlerini doldur
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Sex'i encode et
    le_sex = LabelEncoder()
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    
    # Embarked'ı encode et
    le_embarked = LabelEncoder()
    df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])
    
    # Yeni özellikler
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Model için özellik seçimi
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone']
    
    X = df[features]
    y = df['Survived']
    
    return X, y, features

if __name__ == "__main__":
    # Veri oluştur ve kaydet
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Ham veriyi oluştur
    df = load_titanic_data()
    df.to_csv('data/raw/titanic.csv', index=False)
    print("✅ Ham veri kaydedildi: data/raw/titanic.csv")
    
    # İşlenmiş veriyi hazırla
    X, y, features = prepare_data(df)
    
    # Birleştir ve kaydet
    processed_df = X.copy()
    processed_df['Survived'] = y
    processed_df.to_csv('data/processed/titanic_processed.csv', index=False)
    
    print("✅ İşlenmiş veri kaydedildi: data/processed/titanic_processed.csv")
    print(f"Özellikler: {features}")
    print(f"Veri boyutu: {X.shape}")
    print(f"Hedef dağılımı:\n{y.value_counts()}")