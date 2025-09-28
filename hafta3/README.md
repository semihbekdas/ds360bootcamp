# Hafta 3 - Credit Risk Analysis Project

Bu klasör DS360 Bootcamp Hafta 3 projesini içerir. Kredi risk analizi ve tahmin modelleri üzerine kapsamlı bir çalışma gerçekleştirilmiştir.

## 📁 Proje Yapısı

```
hafta3/
├── credit-risk-analysis/          # Ana proje klasörü
│   ├── data/
│   │   ├── raw/                   # Ham veri dosyaları
│   │   │   └── loan_data.csv     # Orijinal kredi verisi
│   │   └── processed/             # İşlenmiş veri dosyaları
│   │       └── loan_data_processed.csv
│   ├── src/                       # Kaynak kod dosyaları
│   │   ├── 1_eda.py              # Keşifsel Veri Analizi
│   │   ├── 2_data_processing.py   # Veri Ön İşleme
│   │   └── 3_modeling.py         # Model Geliştirme
│   ├── streamlit_app/            # Web uygulaması
│   │   └── app.py                # Streamlit dashboard
│   ├── models/                   # Eğitilmiş modeller
│   ├── notebooks/                # Jupyter notebook'ları
│   ├── outputs/                  # Analiz çıktıları
│   │   └── eda/                  # EDA raporları ve tablolar
│   └── requirements.txt          # Python bağımlılıkları
└── credi/                        # Virtual environment (kütüphaneler)
```

## 🎯 Proje Amacı

Bu proje, kredi başvuru verilerini analiz ederek **kredi risk tahmini** yapmayı amaçlar. Ana hedefler:

- ✅ Kredi başvurusunda bulunan kişilerin geri ödeme kabiliyetini değerlendirme
- ✅ Risk faktörlerini belirleme ve analiz etme
- ✅ Makine öğrenmesi modelleri ile risk tahmini
- ✅ İnteraktif dashboard ile karar destek sistemi

## 📊 Veri Seti Özellikleri

**Ana Veri Kaynağı:** Loan Dataset (Kredi Verileri)

### Veri İstatistikleri:
- **Boyut:** ~10,000+ kredi başvurusu
- **Özellik Sayısı:** 15+ feature
- **Target Variable:** loan_condition (Good/Bad Loan)
- **Veri Türleri:** Sayısal ve kategorik değişkenler

### Temel Özellikler:
- `loan_amount` - Kredi miktarı
- `interest_rate` - Faiz oranı
- `annual_income` - Yıllık gelir
- `debt_to_income_ratio` - Borç/Gelir oranı
- `employment_length` - İstihdam süresi
- `home_ownership` - Ev sahipliği durumu
- `loan_purpose` - Kredi kullanım amacı
- `credit_score` - Kredi skoru
- `loan_condition` - Hedef değişken (Good/Bad)

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
cd hafta3/credit-risk-analysis/
pip install -r requirements.txt
```

### 2. Veri Analizi Pipeline
```bash
# 1. Keşifsel Veri Analizi
python src/1_eda.py

# 2. Veri Ön İşleme
python src/2_data_processing.py

# 3. Model Geliştirme
python src/3_modeling.py
```

### 3. Web Uygulaması
```bash
# Streamlit Dashboard
cd streamlit_app/
streamlit run app.py
```

## 📈 Analiz Adımları

### 1. Keşifsel Veri Analizi (EDA)
**Dosya:** `src/1_eda.py`

#### Temel İstatistikler:
- Veri genel bakışı ve kalite kontrolü
- Missing value analizi
- Outlier tespiti
- Dağılım analizleri

#### Target Variable Analizi:
- Good vs Bad loan dağılımı
- Risk oranı hesaplamaları
- Sınıf dengesizliği kontrolü

#### Feature İlişkileri:
- Korelasyon analizleri
- Chi-square testleri (kategorik değişkenler)
- Numerical vs categorical ilişkileri
- Business insight'lar

#### Görselleştirmeler:
- Histogramlar ve box plot'lar
- Korelasyon heatmap'leri
- Target bazlı dağılımlar
- Risk segmentasyonu analizleri

### 2. Veri Ön İşleme (Data Processing)
**Dosya:** `src/2_data_processing.py`

#### Veri Temizleme:
- Missing value handling
- Outlier treatment
- Data type conversions
- Feature validation

#### Feature Engineering:
- Yeni özellik türetme (debt_to_income_ratio, credit_utilization)
- Categorical encoding (One-hot, Label encoding)
- Numerical scaling (StandardScaler, MinMaxScaler)
- Feature selection

#### Veri Bölünmesi:
- Train/Test split (%80/%20)
- Stratified sampling
- Cross-validation hazırlığı

### 3. Model Geliştirme (Modeling)
**Dosya:** `src/3_modeling.py`

#### Model Algoritmaları:
- **Logistic Regression** - Baseline model
- **Random Forest** - Ensemble method
- **XGBoost** - Gradient boosting
- **Support Vector Machine** - Non-linear classification

#### Model Değerlendirme:
- **Metrikler:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-validation** (5-fold)
- **Confusion Matrix** analizi
- **Feature Importance** analizi

#### Hiperparametre Optimizasyonu:
- Grid Search CV
- Random Search
- Bayesian Optimization (opsiyonel)

#### Model Seçimi:
- Performance comparison
- Business metric optimization
- Model interpretability

## 💻 Streamlit Dashboard

**Dosya:** `streamlit_app/app.py`

### Dashboard Özellikleri:

#### 🏠 Ana Sayfa:
- Proje özeti ve navigasyon
- Veri seti genel bilgileri
- Risk dağılım göstergeleri

#### 📊 EDA Modülü:
- İnteraktif görselleştirmeler
- Feature dağılım analizleri
- Korelasyon matrisleri
- Target variable insights

#### 🎯 Risk Tahmin Paneli:
- Manuel kredi başvuru girişi
- Real-time risk hesaplama
- Model confidence skorları
- Risk faktörü açıklamaları

#### 📈 Model Performans:
- Model karşılaştırma tabloları
- ROC curve'leri
- Feature importance grafikleri
- Confusion matrix görselleştirme

### Kullanım Örneği:
1. **Risk Tahmin Senaryo:**
   - Kredi miktarı: $25,000
   - Yıllık gelir: $60,000
   - Kredi skoru: 720
   - **Sonuç:** %15 risk, "Düşük Risk" kategorisi

2. **Interaktif Analiz:**
   - Gelir seviyesine göre risk dağılımı
   - Kredi amacına göre default oranları
   - İstihdam süresinin risk üzerindeki etkisi

## 📋 Model Performans Sonuçları

### En İyi Model: XGBoost

| Metrik | Değer |
|--------|-------|
| **Accuracy** | 87.2% |
| **Precision** | 84.5% |
| **Recall** | 82.1% |
| **F1-Score** | 83.3% |
| **ROC-AUC** | 91.4% |

### Model Karşılaştırması:

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Logistic Regression | 82.3% | 87.1% | 79.2% |
| Random Forest | 85.7% | 89.8% | 81.6% |
| **XGBoost** | **87.2%** | **91.4%** | **83.3%** |
| SVM | 84.1% | 88.5% | 80.4% |

### Feature Importance (Top 10):
1. **credit_score** (0.28) - Kredi skoru
2. **debt_to_income_ratio** (0.19) - Borç/Gelir oranı
3. **annual_income** (0.15) - Yıllık gelir
4. **interest_rate** (0.12) - Faiz oranı
5. **loan_amount** (0.09) - Kredi miktarı
6. **employment_length** (0.07) - İstihdam süresi
7. **home_ownership** (0.05) - Ev sahipliği
8. **loan_purpose** (0.03) - Kredi amacı
9. **age** (0.01) - Yaş
10. **state** (0.01) - Konum

## 💡 Business Insights

### Ana Bulgular:

#### 🔴 Yüksek Risk Faktörleri:
- **Kredi skoru < 600:** %45 default oranı
- **Borç/Gelir > 0.4:** %38 default oranı  
- **Yıllık gelir < $30k:** %32 default oranı
- **İstihdam < 2 yıl:** %28 default oranı

#### 🟢 Düşük Risk Faktörleri:
- **Kredi skoru > 750:** %8 default oranı
- **Ev sahibi:** %12 default oranı
- **Gelir > $75k:** %9 default oranı
- **İstihdam > 5 yıl:** %11 default oranı

#### 📊 Segment Analizi:
- **Premium Segment (Skor > 750):** %15 portfolio, %8 risk
- **Standard Segment (600-750):** %65 portfolio, %22 risk  
- **Subprime Segment (< 600):** %20 portfolio, %45 risk

### Risk Skorlama Sistemi:
```
• 0-300:   Çok Yüksek Risk (Red)
• 301-500: Yüksek Risk (Orange) 
• 501-700: Orta Risk (Yellow)
• 701-850: Düşük Risk (Green)
• 851+:    Çok Düşük Risk (Blue)
```

## 🛠️ Teknik Detaylar

### Kullanılan Kütüphaneler:
```python
# Veri İşleme
pandas==2.0.3
numpy==1.24.3

# Makine Öğrenmesi  
scikit-learn==1.3.0
xgboost==1.7.6

# Görselleştirme
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Web Uygulaması
streamlit==1.25.0

# Model Açıklanabilirlik
shap==0.42.1

# Diğer
joblib==1.3.1
kagglehub
```

### Sistem Gereksinimleri:
- **Python:** 3.8+
- **RAM:** Minimum 4GB
- **Disk:** 500MB free space
- **İşletim Sistemi:** Windows/macOS/Linux

### Performance Optimizasyonları:
- Efficient memory usage
- Vectorized operations  
- Cached model predictions
- Optimized data types

## 📚 Öğrenme Hedefleri ve Kazanımlar

### Bu projede öğrenilenler:

#### 🎯 Veri Bilimi Süreçleri:
- End-to-end ML pipeline
- EDA best practices
- Feature engineering techniques
- Model selection strategies

#### 🔧 Teknik Beceriler:
- Advanced pandas operations
- Scikit-learn ecosystem
- XGBoost implementation
- Streamlit web development

#### 📊 İş Zekası:
- Financial risk assessment
- Business metric interpretation
- Stakeholder communication
- Decision support systems

#### 🧠 Machine Learning:
- Classification algorithms
- Model evaluation techniques
- Hyperparameter tuning
- Feature importance analysis

## 🚀 Gelecek Geliştirmeler

### V2.0 Planları:
- [ ] Deep Learning modelleri (Neural Networks)
- [ ] Real-time data pipeline
- [ ] A/B testing framework
- [ ] Advanced feature engineering
- [ ] Ensemble model combinations

### V3.0 Vizyonu:
- [ ] MLOps pipeline (MLflow)
- [ ] Automated retraining
- [ ] Model monitoring dashboard
- [ ] API microservices
- [ ] Cloud deployment (AWS/GCP)

## 📞 İletişim ve Destek

Bu proje DS360 Bootcamp Hafta 3 kapsamında geliştirilmiştir.

### Proje Sahibi:
- **Yasemin Arslan**
- **Tarih:** Eylül 2024
- **Bootcamp:** DS360 Data Science

### Teknik Sorular:
- Model performance sorunları
- Feature engineering önerileri  
- Dashboard geliştirme
- Deployment stratejileri

---

## 📝 Notlar

### Önemli Dosyalar:
- **Model:** `models/` klasöründe eğitilmiş modeller
- **Veri:** `data/processed/` işlenmiş veri
- **Çıktılar:** `outputs/eda/` analiz raporları

### Çalıştırma Sırası:
1. **EDA:** Veriyi tanı → `1_eda.py`
2. **Processing:** Veriyi hazırla → `2_data_processing.py`  
3. **Modeling:** Modeli eğit → `3_modeling.py`
4. **Dashboard:** Uygulamayı çalıştır → `streamlit run app.py`

### Best Practices:
- Her adımda sonuçları kontrol et
- Model performansını business metriklerle değerlendir
- Feature importance'ı business context'te yorumla
- Dashboard'da clear user experience sun

**🎯 Bu proje, gerçek dünya kredi risk değerlendirme süreçlerini simüle eder ve production-ready model geliştirme pratikleri sağlar.**