# 📖 Veri Seti Hikayesi: Loan Risk Analysis

## 🏦 Senaryo

Bir fintech şirketi olan **FinanceFlow**, kredi başvurularını değerlendirmek için makine öğrenmesi tabanlı bir risk analiz sistemi geliştirmek istiyor. Şirket, manuel kredi değerlendirme sürecini otomatikleştirerek:

- ⚡ Daha hızlı karar verme
- 📊 Objektif risk değerlendirmesi  
- 💰 Kredi kayıplarını minimize etme
- 🎯 Doğru müşteri segmentasyonu

hedeflerine ulaşmak istiyor.

## 📊 Veri Seti: "Loan Data"

**Kaynak**: Kaggle - zhijinzhai/loandata

### 👥 Veri Hikayesi
Bu veri seti, bir ABD merkezli kredi şirketinin 2007-2018 yılları arasındaki **gerçek kredi başvuru verilerini** içeriyor. Her satır bir kredi başvurusunu temsil ediyor ve şu bilgileri içeriyor:

### 📝 Temel Özellikler

#### 👤 **Demografik Bilgiler**
- **age**: Başvuru sahibinin yaşı
- **employment_length**: İş deneyimi süresi (yıl)
- **home_ownership**: Ev sahipliği durumu (rent/own/mortgage)

#### 💰 **Finansal Bilgiler**  
- **income**: Yıllık gelir ($)
- **loan_amount**: Talep edilen kredi miktarı ($)
- **credit_score**: Kredi skoru (300-850 arası)

#### 🎯 **Kredi Detayları**
- **loan_purpose**: Kredi kullanım amacı
  - `home`: Ev kredisi
  - `auto`: Araba kredisi  
  - `education`: Eğitim kredisi
  - `business`: İş kredisi
  - `personal`: Kişisel kredi

#### ⚠️ **Target Değişken**
- **default**: Kredi geri ödememe durumu
  - `0`: Kredi düzenli ödenmiş (İyi müşteri)
  - `1`: Kredi ödenememiş (Riskli müşteri)

## 🎭 Gerçek Dünya Senaryosu

### 📈 İş Problemi
1. **Dengesiz Veri**: Gerçek hayatta kredilerin %85-90'ı düzenli ödenir
2. **Yanlış Pozitif Maliyeti**: İyi müşteriyi reddetmek → gelir kaybı
3. **Yanlış Negatif Maliyeti**: Kötü müşteriyi kabul etmek → kredi kaybı
4. **Düzenleyici Uyum**: Adil kredi politikaları gerekiyor

### 🔍 Analiz Hedefleri

#### 1. **Keşifsel Veri Analizi (EDA)**
- Risk faktörlerini keşfetmek
- Müşteri segmentlerini anlamak
- Veri kalitesi problemlerini tespit etmek

#### 2. **Dengesiz Veri Problemi Çözümü**
- **SMOTE**: Azınlık sınıfı (default=1) için sentetik örnekler oluştur
- **Undersampling**: Çoğunluk sınıfını (default=0) azalt  
- **Class Weights**: Model eğitiminde sınıf ağırlıklarını ayarla

#### 3. **Model Karşılaştırması**
- **Logistic Regression**: Basit, yorumlanabilir, hızlı
- **XGBoost**: Güçlü, ensemble metodu, yüksek performans

## 🏆 Başarı Kriterleri

### 📊 Model Metrikleri
- **AUC Score**: Ana performans metriği (0.80+ hedef)
- **Precision**: Riskli dediğimiz müşterilerin ne kadarı gerçekten riskli?
- **Recall**: Gerçek riskli müşterilerin ne kadarını yakalıyoruz?

### 💼 İş Metrikleri  
- **False Positive Rate**: İyi müşteri kaybı ≤ %15
- **False Negative Rate**: Kötü kredi kabulü ≤ %20
- **Model Yorumlanabilirliği**: Risk faktörleri açık olmalı

## 🎯 Proje Değeri

### 📚 **Eğitim Açısından**
Bu proje öğrencilere şunları öğretir:
- Gerçek dünya veri problemleri
- Dengesiz veri çözümleri
- Model karşılaştırma teknikleri
- End-to-end ML pipeline
- Streamlit ile dashboard geliştirme
- Cloud deployment

### 🏢 **İş Açısından**
- Risk değerlendirme süresini 2 hafta → 2 dakikaya düşürür
- Manuel hataları %80 azaltır
- Kredi portföy kalitesini %15 iyileştirir
- Müşteri deneyimini dramatik olarak geliştirir

## ⚠️ **Etik Hususlar**

### 🛡️ **Bias Prevention**
- Yaş, cinsiyet gibi korumalı özellikler kullanılmıyor
- Model adaletliliği düzenli test ediliyor
- Şeffaf karar verme süreci

### 🔒 **Veri Güvenliği**
- Kişisel veriler anonimleştirilmiş
- GDPR/KVKK uyumlu süreçler
- Güvenli model deployment

## 🚀 **Sonuç**

Bu proje, öğrencilere **gerçek dünya makine öğrenmesi problemlerini** deneyimleme fırsatı veriyor. Sadece teknik beceriler değil, aynı zamanda:

- İş problemi anlama
- Veri hikayesi çıkarma  
- Etik ML uygulamaları
- Production deployment

konularında da deneyim kazandırıyor.

---

*Bu hikaye, öğrencilerin veri bilimi projelerini sadece teknik egzersiz olarak değil, gerçek iş problemleri olarak görmelerini sağlamak için yazılmıştır.*