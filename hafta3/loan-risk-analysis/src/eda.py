import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

def perform_eda():
    """
    Keşifsel veri analizi yapar
    """
    print("🔍 EDA başlatılıyor...")
    
    # Veri yükleme
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Temel bilgiler
    print("\n📊 Temel Bilgiler:")
    print("Veri Tipleri:")
    print(df.dtypes)
    print("\nEksik Değerler:")
    print(df.isnull().sum())
    print("\nTemel İstatistikler:")
    print(df.describe())
    
    # Target değişken analizi
    print("\n🎯 Target Değişken Analizi:")
    target_cols = [col for col in df.columns if any(word in col.lower() for word in ['default', 'status', 'target', 'class'])]
    if target_cols:
        target_col = target_cols[0]
        print(f"Target column: {target_col}")
        print(df[target_col].value_counts())
        
        # Target dağılımı görselleştirme
        plt.figure(figsize=(8, 5))
        df[target_col].value_counts().plot(kind='bar')
        plt.title('Target Değişken Dağılımı')
        plt.xticks(rotation=0)
        plt.show()
    else:
        print("Target column bulunamadı. Tüm kolonlar:")
        print(df.columns.tolist())
        # Binary olan ilk kolonu target olarak kabul et
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                print(f"Binary target olarak {col} seçildi")
                break
    
    # Sayısal değişken analizi
    print("\n📈 Sayısal Değişken Analizi:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Sayısal kolonlar: {numeric_cols}")
    
    if len(numeric_cols) > 0:
        # Dağılımları görselleştir
        n_cols = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            df[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'{col} Dağılımı')
        
        # Boş subplotları gizle
        for i in range(n_cols, 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    # Kategorik değişken analizi
    print("\n📋 Kategorik Değişken Analizi:")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Kategorik kolonlar: {categorical_cols}")
    
    for col in categorical_cols[:4]:  # İlk 4 kategorik değişken
        print(f"\n{col} dağılımı:")
        print(df[col].value_counts())
        
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'{col} Dağılımı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Korelasyon analizi
    print("\n🔗 Korelasyon Analizi:")
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Korelasyon Matrisi')
        plt.tight_layout()
        plt.show()
        
        # En yüksek korelasyonları bul
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # 0.5'ten yüksek korelasyonlar
                    corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        if corr_pairs:
            print("Yüksek korelasyonlu değişken çiftleri:")
            for col1, col2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"{col1} - {col2}: {corr:.3f}")
    
    # Target vs Features analizi
    if 'target_col' in locals() and len(numeric_cols) > 1:
        print("\n🎯 Target vs Features Analizi:")
        numeric_features = [col for col in numeric_cols if col != target_col][:4]
        
        if numeric_features:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_features):
                if i < 4:
                    df.boxplot(column=col, by=target_col, ax=axes[i])
                    axes[i].set_title(f'{col} by {target_col}')
                    axes[i].set_xlabel(target_col)
            
            plt.tight_layout()
            plt.show()
    
    # Veri kalitesi kontrolü
    print("\n🔍 Veri Kalitesi Kontrolü:")
    print(f"Toplam satır sayısı: {len(df)}")
    print(f"Toplam kolon sayısı: {len(df.columns)}")
    print(f"Eksik değer oranı: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    print(f"Duplicate satır sayısı: {df.duplicated().sum()}")
    
    # Aykırı değer kontrolü (basit IQR yöntemi)
    print("\nAykırı Değer Analizi (IQR):")
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(df) * 100
        outlier_summary.append((col, outlier_count, outlier_pct))
        print(f"{col}: {outlier_count} aykırı değer ({outlier_pct:.1f}%)")
    
    print("\n✅ EDA tamamlandı!")
    return df, outlier_summary

if __name__ == "__main__":
    df, outlier_summary = perform_eda()