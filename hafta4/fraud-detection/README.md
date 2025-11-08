# Hafta 4: Finans Sprinti - Fraud Detection 🏦

## 📋 Konu Başlıkları ve Öğrenilecekler

### 1. 🎯 Outlier Detection
- **Isolation Forest**: Anomaly detection için ensemble method
- **Local Outlier Factor (LOF)**: Density-based outlier detection
- **Karşılaştırma**: Supervised vs Unsupervised yaklaşımlar
- **Hyperparameter tuning**: contamination, n_neighbors optimizasyonu

### 2. 🔧 Feature Scaling ve Encoding
- **Scaling Methods**: StandardScaler vs RobustScaler vs MinMaxScaler
- **Categorical Encoding**: OneHot vs Label vs Ordinal
- **Feature Engineering**: Log transformation, interaction features
- **Missing Value Handling**: Imputation strategies
- **Class Imbalance**: SMOTE, ADASYN, SMOTETomek

### 3. 📊 ROC-AUC ve PR-AUC Metrikleri
- **ROC-AUC**: True Positive Rate vs False Positive Rate
- **PR-AUC**: Precision vs Recall (imbalanced data için kritik)
- **Threshold Optimization**: Business cost minimization
- **Confusion Matrix**: TP, FP, TN, FN analizi
- **Business Metrics**: Cost-benefit analysis

### 4. 🔍 Model Açıklanabilirlik
- **SHAP (SHapley Additive exPlanations)**:
  - TreeExplainer: Tree-based modeller için
  - KernelExplainer: Model-agnostic yaklaşım
  - Feature importance ve dependence plots
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Local açıklamalar
  - Tabular explainer
- **Global vs Local açıklamalar**
- **Fraud pattern analizi**

### 5. 🚀 CI/CD Pipeline ve Deployment
- **Data Validation**: Schema ve quality checks
- **Model Training**: Automated retraining
- **Performance Monitoring**: Drift detection
- **A/B Testing**: Model comparison
- **Production Deployment**: Staging → Production
- **Security**: Model signing, access control

## 🎯 Kullanılan Dataset: Credit Card Fraud Detection

Bu proje **gerçek dünya fraud detection** problemini simüle etmek için en uygun dataset'i kullanır:

### Dataset Özellikleri
- **Boyut**: 284,807 işlem
- **Imbalance**: %99.83 Normal, %0.17 Fraud (gerçekçi oran)
- **Features**: 30 kolon
  - `V1-V28`: PCA ile dönüştürülmüş gizli features (privacy için)
  - `Time`: İşlem zamanı (saniye cinsinden)
  - `Amount`: İşlem tutarı
  - `Class`: 0=Normal, 1=Fraud (target variable)

### Neden Bu Dataset?
- ✅ **Gerçekçi imbalance**: Production fraud rate'ini yansıtır
- ✅ **Preprocessed**: PCA ile privacy korunmuş
- ✅ **Temizlenmiş**: Missing value yok
- ✅ **Hızlı training**: Makul boyut
- ✅ **Eğitim dostu**: Tüm konular için ideal

## 🏗️ Proje Yapısı

```
hafta4/fraud-detection/
├── src/                           # Ana kaynak kodlar
│   ├── outlier_detection.py       # Isolation Forest & LOF
│   ├── preprocessing.py           # Feature scaling & encoding
│   ├── evaluation.py             # ROC-AUC, PR-AUC metrikleri
│   ├── explainability_clean.py   # SHAP/LIME açıklamaları
│   └── pipeline.py               # End-to-end pipeline
├── tests/                         # Unit testler
│   └── test_pipeline.py          
├── .github/workflows/             # CI/CD pipeline
│   └── ci_cd.yml                 # GitHub Actions
├── config/                        # Konfigürasyon dosyaları
│   └── config.yaml               # Model ve training parametreleri
├── data/                          # Veri klasörleri
│   ├── raw/                      # Ham veri
│   └── processed/                # İşlenmiş veri
├── models/                        # Eğitilmiş modeller
├── logs/                          # Log dosyaları
├── notebooks/                     # Jupyter notebooks (analiz için)
├── download_data.py              # Dataset indirme utility
├── run_demo.py                   # Interaktif demo
├── requirements.txt              # Python dependencies
└── README.md                     # Bu dosya
```

## ⚡️ Hızlı Başlangıç

```bash
# Proje klasörüne geç
cd hafta4/fraud-detection

# Sanal ortam (önerilir)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

Pipeline varsayılan olarak synthetic fraud datası üretir. Gerçek dataset kullanmak istersen KaggleHub ile indirme seçeneği mevcut.

```bash
# Varsayılan: synthetic data, modeller kaydedilir
python src/pipeline.py --mode train --save_models

# Kaggle dataset'ini otomatik indirip eğit
python src/pipeline.py --mode train --use_kagglehub --save_models

# Belirli bir CSV ile çalış
python src/pipeline.py --mode train --data data/raw/creditcard.csv --save_models
```

CLI modları:
- `--mode train` (varsayılan): End-to-end pipeline, değerlendirme ve açıklanabilirlik.
- `--mode predict`: Kayıtlı modellerle hızlı tahmin çıktısı gösterir.
- `--mode explain`: Seçilen model için SHAP/LIME özetleri üretir.

`--load_models` bayrağı hazır modelleri `models/` klasöründen yükler. Modeller yoksa pipeline otomatik yeniden eğitir.

## 🔧 Veri Yönetimi

```bash
# Kaggle datasını helper script ile indir
python src/1_download_data.py

# Demo (synthetic data ile menü tabanlı)
python run_demo.py
```

Pipeline çıktıları:
- `models/` içinde `*_model.pkl`, `preprocessor.pkl`, `feature_info.pkl`
- `mlruns/` altında MLflow metadata (varsayılan olarak yerel SQLite backend)
- `logs/` klasörü altında `fraud_detection.log`

SHAP waterfall grafiği bazı kombinasyonlarda `Explanation` nesnesi beklediği için uyarı verebilir; pipeline akışı bu uyarıyı loglayıp devam eder.

MLflow için özel bir sunucu belirtmediysen config fallback olarak `mlruns/mlflow.db` SQLite dosyasını kullanır. UI görmek için:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## 🎮 Demo ve Örnekler

### 1. İnteraktif Demo
```bash
python run_demo.py
```
**Menu seçenekleri:**
- Preprocessing demo
- Outlier detection demo  
- Evaluation metrikleri
- Model açıklanabilirlik
- Full pipeline

### 2. Outlier Detection
```python
from src.outlier_detection import OutlierDetector

# Isolation Forest
detector = OutlierDetector(contamination=0.002)  # %0.2 fraud oranı
detector.fit_isolation_forest(X_train_processed.values)
labels, scores = detector.predict_isolation_forest(X_test_processed.values)

# Performance evaluation
metrics = detector.evaluate_performance(y_test_processed, scores)
print(metrics)
```

### 3. Feature Preprocessing
```python
from src.preprocessing import FeaturePreprocessor, ImbalanceHandler

# Preprocessing
preprocessor = FeaturePreprocessor(
    scaling_method='robust',  # Outlier'lara dayanıklı
    encoding_method='onehot'
)
processed = preprocessor.fit_transform(train_df, target_col='target')
X_train_processed = processed.drop('target', axis=1)
y_train_processed = processed['target']

# Imbalance handling
X_balanced, y_balanced = ImbalanceHandler.apply_smote(X_train_processed, y_train_processed)
```

### 4. Model Evaluation
```python
from src.evaluation import FraudEvaluator

evaluator = FraudEvaluator(model, "Random Forest")
results = evaluator.evaluate_binary_classification(X_test_processed, y_test_processed)
fpr, tpr, roc_thresholds = evaluator.roc_curve_points()
precision, recall, pr_thresholds = evaluator.precision_recall_points()
```

### 5. Model Açıklanabilirlik
```python
import numpy as np
from src.explainability_clean import ModelExplainer

explainer = ModelExplainer(
  model,
  X_balanced,
  feature_names=list(X_train_processed.columns),
  class_names=['Normal', 'Fraud'],
  y_train=y_balanced,
)

X_sample = X_test_processed.head(100)
if explainer.initialize_shap('tree'):
  shap_values, X_sample = explainer.compute_shap_values(X_sample)
  if shap_values is not None:
    explainer.plot_shap_summary(X_sample)
    explainer.plot_shap_waterfall(X_sample, index=0)

importance = explainer.global_feature_importance(X_sample, y_test_processed[:len(X_sample)])
fraud_patterns = explainer.analyze_fraud_patterns(np.asarray(X_sample), y_test_processed[:len(X_sample)])
```

## ✅ Testler

Pipeline değişikliklerini kontrol etmek için unit testler mevcut. `pytest` varsayılan gereksinim listesinde değildir; manuel kurulumdan sonra testleri çalıştırabilirsin.

```bash
pip install pytest
python -m pytest tests/test_simple.py tests/test_pipeline.py
```

Testler SMOTE ve SHAP gibi bileşenleri kullandığı için ilk çalıştırmada modellerin veya preprocess dosyalarının oluşturulmuş olması gerekir (`python src/pipeline.py --mode train --save_models`).

## 📊 Beklenen Sonuçlar ve Öğrenim Hedefleri

### 1. Outlier Detection Performance
- **Isolation Forest**: ROC-AUC ~0.85-0.90
- **LOF**: ROC-AUC ~0.80-0.85
- **Hybrid approach**: Ensemble of methods

### 2. Supervised Learning Benchmarks
- **Random Forest**: ROC-AUC ~0.95+, PR-AUC ~0.75+
- **Logistic Regression**: ROC-AUC ~0.93+, PR-AUC ~0.65+
- **Gradient Boosting**: ROC-AUC ~0.96+, PR-AUC ~0.80+

### 3. Feature Importance Insights
- **Time patterns**: Gece fraud'ları daha yüksek
- **Amount patterns**: Küçük ve çok büyük miktarlar risky
- **PCA features**: V4, V11, V12 genellikle önemli

### 4. Business Metrics
- **Optimal threshold**: ~0.3-0.4 (cost minimization için)
- **Cost reduction**: %60-80 compared to random checking
- **False positive rate**: <%5 (customer experience için)

## 🎯 Learning Path ve Exercises

### Başlangıç Seviyesi
1. **Dataset exploration**: EDA ve basic statistics
2. **Simple outlier detection**: Isolation Forest ile başla
3. **Basic preprocessing**: Scaling ve encoding
4. **Model training**: Single algorithm (Random Forest)
5. **Basic evaluation**: ROC-AUC ve confusion matrix

### Orta Seviye
1. **Multiple outlier methods**: IF + LOF karşılaştırması
2. **Advanced preprocessing**: Feature engineering
3. **Imbalance handling**: SMOTE ile class balancing
4. **Threshold optimization**: Business cost minimization
5. **Model comparison**: Multiple algorithms

### İleri Seviye
1. **Ensemble methods**: Multiple outlier detector fusion
2. **Custom preprocessing pipeline**: Domain-specific features
3. **Advanced evaluation**: PR-AUC focus, cost-sensitive metrics
4. **Model explainability**: SHAP + LIME comprehensive analysis
5. **Production pipeline**: CI/CD ile automated deployment

### Expert Seviye
1. **Real-time detection**: Streaming data processing
2. **Model drift detection**: Performance monitoring
3. **A/B testing**: Model comparison in production
4. **Custom explainability**: Domain-specific explanations
5. **End-to-end MLOps**: Complete production system

## 🔥 Advanced Features

### 1. Hyperparameter Optimization
```python
# Grid search for optimal parameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'contamination': [0.001, 0.002, 0.005],
    'n_estimators': [50, 100, 200]
}
```

### 2. Ensemble Methods
```python
# Combine multiple outlier detectors
ensemble_prediction = (if_pred + lof_pred + svm_pred) / 3
```

### 3. Real-time Scoring
```python
# Fast inference pipeline
@app.route('/predict', methods=['POST'])
def predict_fraud():
    features = preprocess(request.json)
    prediction = model.predict_proba(features)[0][1]
    return {'fraud_probability': prediction}
```

### 4. Model Monitoring
```python
# Performance drift detection
def detect_drift(model, X_new, threshold=0.1):
    current_auc = roc_auc_score(y_true, model.predict_proba(X_new)[:, 1])
    return abs(baseline_auc - current_auc) > threshold
```

## 🚀 Production Deployment

### 1. CI/CD Pipeline
```yaml
# .github/workflows/ci_cd.yml
- Data validation
- Model training  
- Performance testing
- Security scanning
- Staging deployment
- Production deployment
- Monitoring setup
```

### 2. Model Serving
```python
# FastAPI ile REST API
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict_transaction(transaction: TransactionModel):
    # Preprocessing + Prediction + Logging
    return PredictionResponse(is_fraud=prediction, confidence=confidence)
```

### 3. Monitoring Dashboard
- **Performance metrics**: ROC-AUC, PR-AUC trends
- **Business metrics**: False positive rate, cost savings
- **Data drift**: Feature distribution changes
- **Model drift**: Performance degradation alerts

## 🎓 Öğrenim Çıktıları

Bu projeyi tamamladıktan sonra şunları öğreneceksin:

### Teknik Skills
- ✅ Outlier detection algorithms (IF, LOF)
- ✅ Feature engineering for fraud detection
- ✅ Imbalanced learning techniques (SMOTE, cost-sensitive)
- ✅ Model evaluation for imbalanced problems
- ✅ Explainable AI (SHAP, LIME)
- ✅ MLOps pipeline (CI/CD, monitoring)

### Business Understanding
- ✅ Fraud detection domain knowledge
- ✅ Cost-benefit analysis
- ✅ Threshold optimization
- ✅ False positive vs false negative trade-offs
- ✅ Real-time vs batch processing decisions

### Production Skills
- ✅ Model deployment strategies
- ✅ Performance monitoring
- ✅ A/B testing for models
- ✅ Security considerations
- ✅ Scalability planning

## 🤝 Katkıda Bulunma

Bu eğitim materyalini geliştirmek için:
1. Issues açabilirsin
2. Pull request gönderebilirsin  
3. Yeni dataset önerileri yapabilirsin
4. Documentation iyileştirmeleri yapabilirsin

## 📚 Ek Kaynaklar

### Fraud Detection
- [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- [Imbalanced Learning](https://imbalanced-learn.org/)

### Explainable AI
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Tutorial](https://github.com/marcotcr/lime)

### MLOps
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLOps Best Practices](https://ml-ops.org/)

---

**🎯 Ready to start? Run `python run_demo.py` and begin your fraud detection journey!**