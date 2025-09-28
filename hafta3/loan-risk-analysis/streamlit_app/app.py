import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

# Add src directory to path
sys.path.append('../src')

st.set_page_config(
    page_title="Loan Risk Dashboard",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model_and_preprocessor():
    """
    Model ve preprocessor'u yükler
    """
    try:
        # Model dosyalarını kontrol et
        model_files = []
        if os.path.exists('../models'):
            model_files = [f for f in os.listdir('../models') if f.endswith('.pkl')]
        
        if not model_files:
            return None, None, "Model dosyaları bulunamadı. Lütfen önce modelleri eğitin."
        
        # En son model dosyasını yükle (alfabetik olarak son)
        model_file = sorted(model_files)[-1]
        model_path = os.path.join('../models', model_file)
        model = joblib.load(model_path)
        
        return model, None, f"Model yüklendi: {model_file}"
    except Exception as e:
        return None, None, f"Model yüklenirken hata: {str(e)}"

def main():
    st.title("💰 Loan Risk Assessment Dashboard")
    st.markdown("Bu dashboard ile kredi başvurularının risk analizi yapabilirsiniz.")
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    
    # Model yükle
    model, preprocessor, status = load_model_and_preprocessor()
    st.sidebar.info(status)
    
    # Ana içerik
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Tahmini", "📊 Model Performansı", "📈 Veri Analizi"])
    
    with tab1:
        st.header("Kredi Risk Tahmini")
        
        if model is None:
            st.error("Model yüklenemedi. Lütfen önce modelleri eğitin.")
            st.code("""
            # Modelleri eğitmek için:
            cd src
            python models.py
            """)
            return
        
        # Input formu
        st.subheader("Başvuru Bilgileri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Yaş", 18, 80, 35)
            income = st.number_input("Yıllık Gelir ($)", 
                                   min_value=10000, max_value=500000, value=50000, step=5000)
            employment_length = st.slider("İş Deneyimi (yıl)", 0, 40, 5)
            
        with col2:
            loan_amount = st.number_input("Kredi Miktarı ($)", 
                                        min_value=1000, max_value=200000, value=25000, step=1000)
            credit_score = st.slider("Kredi Skoru", 300, 850, 650)
            
        col3, col4 = st.columns(2)
        with col3:
            loan_purpose = st.selectbox("Kredi Amacı", 
                                      ['home', 'auto', 'education', 'business', 'personal'])
        with col4:
            home_ownership = st.selectbox("Ev Sahipliği", 
                                        ['rent', 'own', 'mortgage'])
        
        # Tahmin butonu
        if st.button("🎯 Risk Tahmini Yap", type="primary"):
            try:
                # Input verilerini hazırla
                input_data = pd.DataFrame({
                    'age': [age],
                    'income': [income],
                    'loan_amount': [loan_amount],
                    'employment_length': [employment_length],
                    'credit_score': [credit_score],
                    'loan_purpose': [loan_purpose],
                    'home_ownership': [home_ownership]
                })
                
                # Basit encoding (production'da preprocessor kullanılmalı)
                purpose_map = {'home': 0, 'auto': 1, 'education': 2, 'business': 3, 'personal': 4}
                ownership_map = {'rent': 0, 'own': 1, 'mortgage': 2}
                
                input_data['loan_purpose'] = input_data['loan_purpose'].map(purpose_map)
                input_data['home_ownership'] = input_data['home_ownership'].map(ownership_map)
                
                # Tahmin yap
                if hasattr(model, 'predict_proba'):
                    risk_prob = model.predict_proba(input_data)[0][1]
                    risk_prediction = model.predict(input_data)[0]
                else:
                    risk_prediction = model.predict(input_data)[0]
                    risk_prob = risk_prediction
                
                # Sonuçları göster
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if risk_prediction == 1:
                        st.error("🚨 YÜKSEK RİSK")
                        st.markdown("Bu başvuru **riskli** olarak değerlendirildi.")
                    else:
                        st.success("✅ DÜŞÜK RİSK")
                        st.markdown("Bu başvuru **güvenli** olarak değerlendirildi.")
                
                with col_result2:
                    # Risk probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Skoru (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk faktörleri analizi
                st.subheader("📋 Risk Faktörleri")
                
                risk_factors = []
                if income < 30000:
                    risk_factors.append("Düşük gelir seviyesi")
                if credit_score < 600:
                    risk_factors.append("Düşük kredi skoru")
                if loan_amount > income * 3:
                    risk_factors.append("Gelire oranla yüksek kredi miktarı")
                if employment_length < 1:
                    risk_factors.append("Kısa iş deneyimi")
                
                if risk_factors:
                    st.warning("⚠️ Tespit edilen risk faktörleri:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("✅ Önemli risk faktörü tespit edilmedi.")
                
            except Exception as e:
                st.error(f"Tahmin yapılırken hata oluştu: {str(e)}")
    
    with tab2:
        st.header("Model Performans Metrikleri")
        
        # Mock performance data
        st.subheader("Model Karşılaştırması")
        
        model_comparison = pd.DataFrame({
            'Model': ['Logistic Regression (Original)', 'Logistic Regression (SMOTE)', 
                     'XGBoost (Original)', 'XGBoost (Class Weights)'],
            'AUC Score': [0.75, 0.78, 0.82, 0.80],
            'Precision': [0.70, 0.74, 0.79, 0.77],
            'Recall': [0.68, 0.76, 0.78, 0.79]
        })
        
        st.dataframe(model_comparison, use_container_width=True)
        
        # Performance chart
        fig = px.bar(model_comparison, x='Model', y='AUC Score', 
                    title='Model AUC Score Karşılaştırması')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.header("Veri Seti Analizi")
        
        try:
            # Sample data için mock data göster
            st.subheader("Veri Seti Özellikleri")
            
            # Mock data statistics
            stats_data = {
                'Özellik': ['Yaş', 'Gelir', 'Kredi Miktarı', 'İş Deneyimi', 'Kredi Skoru'],
                'Ortalama': [35.2, 65420, 48230, 8.5, 650],
                'Std': [10.5, 25400, 28100, 6.2, 95],
                'Min': [18, 15000, 5000, 0, 350],
                'Max': [75, 180000, 180000, 35, 825]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Default rate visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Mock default rate by age group
                age_default = pd.DataFrame({
                    'Yaş Grubu': ['18-25', '26-35', '36-45', '46-55', '56+'],
                    'Default Oranı': [0.18, 0.15, 0.12, 0.10, 0.08]
                })
                
                fig = px.bar(age_default, x='Yaş Grubu', y='Default Oranı',
                            title='Yaş Grubuna Göre Default Oranı')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Mock default rate by loan purpose
                purpose_default = pd.DataFrame({
                    'Kredi Amacı': ['home', 'auto', 'education', 'business', 'personal'],
                    'Default Oranı': [0.08, 0.12, 0.10, 0.18, 0.15]
                })
                
                fig = px.pie(purpose_default, values='Default Oranı', names='Kredi Amacı',
                           title='Kredi Amacına Göre Default Oranı')
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Veri analizi gösterilirken hata: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Not:** Bu dashboard eğitim amaçlıdır. Gerçek kredi kararları için profesyonel danışmanlık alınmalıdır.")

if __name__ == "__main__":
    main()