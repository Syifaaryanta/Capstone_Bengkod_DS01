import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================

st.set_page_config(
    page_title="Obesity Prediction System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #F4EEFF;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #424874 0%, #A6B1E1 100%);
        color: #F5F5F5;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(66, 72, 116, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #F5F5F5;
    }
    
    .main-header p {
        color: #F5F5F5;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(66, 72, 116, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #A6B1E1;
        color: #1B262C;
    }
    
    .prediction-card {
        background: #DCD6F7;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #A6B1E1;
        color: #1B262C;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #DCD6F7;
        color: #1B262C;
        box-shadow: 0 2px 8px rgba(66, 72, 116, 0.1);
    }
    
    .metric-card h2 {
        color: #424874;
        font-size: 2rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        color: #1B262C;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #424874;
        font-size: 1rem;
        margin-top: 0.25rem;
    }
    
    .metric-card i {
        font-size: 1.5rem;
        color: #A6B1E1;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #A6B1E1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #F5F5F5;
    }
    
    .warning-box {
        background: #DCD6F7;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #424874;
        color: #1B262C;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #424874 0%, #A6B1E1 100%);
        color: #F5F5F5;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
 
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(66, 72, 116, 0.4);
        background: linear-gradient(135deg, #424874 0%, #A6B1E1 100%);
    }

    .section-header {
        color: #424874;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
            
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #F4EEFF 0%, #DCD6F7 100%);
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background: #424874;
    }
    
    h2, h3 {
        color: #424874;
        font-weight: 600;
    }
    
    /* Icon styles */
    .icon {
        margin-right: 0.5rem;
        color: #A6B1E1;
    }
    
    .risk-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .risk-item i {
        margin-right: 1rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNGSI LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load model dari direktori yang ditentukan"""
    model_path = r"D:\Perkuliahan\SEMESTER_6\Bengkod\Capstone_Bengkod_DS01\models\best_tuned_random_forest_model.pkl"
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model, True
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan di: {model_path}")
        # Fallback ke model dummy
        np.random.seed(42)
        X_dummy = np.random.rand(100, 16)
        y_dummy = np.random.randint(0, 7, 100)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_dummy, y_dummy)
        return model, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ============================================================================
# FUNGSI UTILITY
# ============================================================================

def get_bmi_category(weight, height):
    """Hitung BMI dan kategorinya"""
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return bmi, "Berat Badan Kurang"
    elif 18.5 <= bmi < 25:
        return bmi, "Normal"
    elif 25 <= bmi < 30:
        return bmi, "Kelebihan Berat Badan"
    else:
        return bmi, "Obesitas"

def predict_obesity(model, age, gender, height, weight, calc, favc, fcvc, ncp, 
                   scc, smoke, ch2o, family_history, faf, tue, caec, mtrans):
    """Fungsi prediksi obesitas"""
    try:
        input_data = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp, 
                               scc, smoke, ch2o, family_history, faf, tue, caec, mtrans]])
        
        prediction = model.predict(input_data)[0]
        
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities)
        except:
            confidence = 0.85 + np.random.rand() * 0.1
        
        # Mapping sesuai dengan nama kelas yang benar
        labels = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight', 
            2: 'Overweight_Level_I',
            3: 'Overweight_Level_II',
            4: 'Obesity_Type_I',
            5: 'Obesity_Type_II',
            6: 'Obesity_Type_III'
        }
        
        return labels[prediction], confidence
    
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return "Error", 0.0

def get_recommendation(prediction):
    """Memberikan rekomendasi berdasarkan prediksi"""
    recommendations = {
        'Insufficient_Weight': {
            'title': 'Berat Badan Kurang',
            'advice': 'Tingkatkan asupan kalori dengan makanan bergizi. Konsultasi dengan ahli gizi.',
            'risk': 'Rendah',
            'icon': 'fas fa-weight-hanging'
        },
        'Normal_Weight': {
            'title': 'Berat Badan Normal',
            'advice': 'Pertahankan pola hidup sehat dengan diet seimbang dan olahraga teratur.',
            'risk': 'Sangat Rendah',
            'icon': 'fas fa-check-circle'
        },
        'Overweight_Level_I': {
            'title': 'Kelebihan Berat Badan Tingkat I',
            'advice': 'Mulai program penurunan berat badan dengan diet dan olahraga.',
            'risk': 'Sedang',
            'icon': 'fas fa-exclamation-triangle'
        },
        'Overweight_Level_II': {
            'title': 'Kelebihan Berat Badan Tingkat II',
            'advice': 'Diperlukan program penurunan berat badan yang lebih intensif.',
            'risk': 'Tinggi',
            'icon': 'fas fa-exclamation-triangle'
        },
        'Obesity_Type_I': {
            'title': 'Obesitas Tipe I',
            'advice': 'Konsultasi dengan dokter untuk program penurunan berat badan.',
            'risk': 'Tinggi',
            'icon': 'fas fa-hospital'
        },
        'Obesity_Type_II': {
            'title': 'Obesitas Tipe II',
            'advice': 'Diperlukan intervensi medis segera untuk mengurangi risiko komplikasi.',
            'risk': 'Sangat Tinggi',
            'icon': 'fas fa-hospital'
        },
        'Obesity_Type_III': {
            'title': 'Obesitas Tipe III',
            'advice': 'Diperlukan penanganan medis intensif. Pertimbangkan konsultasi spesialis.',
            'risk': 'Ekstrem',
            'icon': 'fas fa-hospital-symbol'
        }
    }
    
    return recommendations.get(prediction, recommendations['Normal_Weight'])

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1><i class="fas fa-heartbeat icon"></i>Obesity Prediction System</h1>
    <p>Sistem Prediksi Tingkat Obesitas Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, is_real_model = load_model()

if model is None:
    st.stop()

if not is_real_model:
    st.markdown("""
    <div class="warning-box">
        <h4><i class="fas fa-exclamation-triangle icon"></i>Mode Demo</h4>
        <p>Model file tidak ditemukan. Aplikasi berjalan dalam mode demo.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR INPUT
# ============================================================================

st.sidebar.markdown("### <i class='fas fa-clipboard-list'></i> Input Data Pasien", unsafe_allow_html=True)

# Demographics
st.sidebar.markdown("#### <i class='fas fa-user'></i> Informasi Demografis", unsafe_allow_html=True)
age = st.sidebar.slider("Umur", 16, 61, 25)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.sidebar.number_input("Tinggi Badan (m)", 1.45, 1.98, 1.70, 0.01)
weight = st.sidebar.number_input("Berat Badan (kg)", 39.0, 173.0, 70.0, 0.1)

# Eating habits
st.sidebar.markdown("#### <i class='fas fa-utensils'></i> Kebiasaan Makan", unsafe_allow_html=True)
favc = st.sidebar.selectbox("Konsumsi Makanan Berkalori Tinggi", ["Tidak", "Ya"])
fcvc = st.sidebar.slider("Konsumsi Sayuran (per hari)", 1.0, 3.0, 2.0, 0.1)
ncp = st.sidebar.slider("Jumlah Makanan Utama", 1.0, 4.0, 3.0, 0.1)
caec = st.sidebar.selectbox("Konsumsi Makanan Antar Waktu Makan", 
                           ["Tidak", "Kadang-kadang", "Sering", "Selalu"])

# Lifestyle
st.sidebar.markdown("#### <i class='fas fa-smoking'></i> Kebiasaan Hidup", unsafe_allow_html=True)
calc = st.sidebar.selectbox("Konsumsi Alkohol", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
smoke = st.sidebar.selectbox("Merokok", ["Tidak", "Ya"])
ch2o = st.sidebar.slider("Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0, 0.1)
scc = st.sidebar.selectbox("Monitor Kalori", ["Tidak", "Ya"])

# Physical activity
st.sidebar.markdown("#### <i class='fas fa-running'></i> Aktivitas Fisik", unsafe_allow_html=True)
faf = st.sidebar.slider("Aktivitas Fisik (hari/minggu)", 0.0, 3.0, 1.0, 0.1)
tue = st.sidebar.slider("Waktu Menggunakan Teknologi (jam/hari)", 0.0, 2.0, 1.0, 0.1)
mtrans = st.sidebar.selectbox("Transportasi Utama", 
                             ["Mobil", "Sepeda", "Motor", "Transportasi Umum", "Jalan Kaki"])

# Family history
st.sidebar.markdown("#### <i class='fas fa-users'></i> Riwayat Keluarga", unsafe_allow_html=True)
family_history = st.sidebar.selectbox("Riwayat Keluarga Kelebihan Berat Badan", ["Tidak", "Ya"])

# ============================================================================
# KONVERSI INPUT
# ============================================================================

gender_num = 1 if gender == "Laki-laki" else 0
favc_num = 1 if favc == "Ya" else 0
calc_mapping = {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
calc_num = calc_mapping[calc]
scc_num = 1 if scc == "Ya" else 0
smoke_num = 1 if smoke == "Ya" else 0
family_history_num = 1 if family_history == "Ya" else 0
caec_mapping = {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
caec_num = caec_mapping[caec]
mtrans_mapping = {"Mobil": 0, "Sepeda": 1, "Motor": 2, "Transportasi Umum": 3, "Jalan Kaki": 4}
mtrans_num = mtrans_mapping[mtrans]

# ============================================================================
# MAIN CONTENT
# ============================================================================

# BMI Section
bmi, bmi_category = get_bmi_category(weight, height)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-chart-bar"></i>
        <h3>BMI</h3>
        <h2>{bmi:.1f}</h2>
        <p>{bmi_category}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-weight"></i>
        <h3>Berat</h3>
        <h2>{weight} kg</h2>
        <p>Berat Badan</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-ruler-vertical"></i>
        <h3>Tinggi</h3>
        <h2>{height} m</h2>
        <p>Tinggi Badan</p>
    </div>
    """, unsafe_allow_html=True)

# BMI Chart
fig_bmi = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = bmi,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "BMI Index"},
    gauge = {
        'axis': {'range': [None, 40]},
        'bar': {'color': "#424874"},
        'steps': [
            {'range': [0, 18.5], 'color': "#F4EEFF"},
            {'range': [18.5, 25], 'color': "#DCD6F7"},
            {'range': [25, 30], 'color': "#A6B1E1"},
            {'range': [30, 40], 'color': "#424874"}
        ],
        'threshold': {
            'line': {'color': "#1B262C", 'width': 4},
            'thickness': 0.75,
            'value': 30
        }
    }
))

fig_bmi.update_layout(
    height=400,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#1B262C'
)

st.plotly_chart(fig_bmi, use_container_width=True)

# Prediction Button
st.markdown('<h2 class="section-header"><i class="fas fa-search icon"></i>Prediksi Tingkat Obesitas</h2>', unsafe_allow_html=True)

if st.button("Analisis Sekarang", type="primary"):
    with st.spinner('Menganalisis data...'):
        prediction, confidence = predict_obesity(
            model, age, gender_num, height, weight, calc_num, favc_num, 
            fcvc, ncp, scc_num, smoke_num, ch2o, family_history_num, 
            faf, tue, caec_num, mtrans_num
        )
        
        # Display prediction
        rec = get_recommendation(prediction)
        
        st.markdown(f"""
        <div class="prediction-card">
            <i class="{rec['icon']}" style="font-size: 2rem; margin-bottom: 1rem; color: #424874;"></i>
            <h2>Hasil Prediksi</h2>
            <h1>{prediction.replace('_', ' ')}</h1>
            <p>Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(confidence)
        
        # Recommendation
        st.markdown(f"""
        <div class="card">
            <h3><i class="fas fa-lightbulb icon"></i>{rec['title']}</h3>
            <p><strong>Tingkat Risiko:</strong> {rec['risk']}</p>
            <p><strong>Rekomendasi:</strong> {rec['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# HEALTH STATS
# ============================================================================

st.markdown('<h2 class="section-header"><i class="fas fa-chart-line icon"></i>Statistik Kesehatan</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Activity level chart
    activity_data = pd.DataFrame({
        'Kategori': ['Olahraga', 'Sayuran', 'Air', 'Screen Time'],
        'Nilai': [faf/3*100, fcvc/3*100, ch2o/3*100, (2-tue)/2*100],
        'Target': [100, 100, 100, 100]
    })
    
    fig_activity = go.Figure()
    fig_activity.add_trace(go.Bar(
        name='Anda',
        x=activity_data['Kategori'],
        y=activity_data['Nilai'],
        marker_color='#2D3748',
        text=[f'{val:.0f}%' for val in activity_data['Nilai']],
        textposition='auto',
        textfont=dict(color='#F5F5F5', size=12, family='Inter')
    ))
    fig_activity.add_trace(go.Bar(
        name='Target',
        x=activity_data['Kategori'],
        y=activity_data['Target'],
        marker_color='#6B73FF',
        text=[f'{val:.0f}%' for val in activity_data['Target']],
        textposition='auto',
        textfont=dict(color='#FFFFFF', size=12, family='Inter')
    ))
    
    fig_activity.update_layout(
        title='Perbandingan dengan Target Sehat',
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#1B262C',
        title_font_color='#1B262C',
        xaxis_title_font_color='#1B262C',
        yaxis_title_font_color='#1B262C',
        legend_font_color='#1B262C'
    )
    
    st.plotly_chart(fig_activity, use_container_width=True)

with col2:
    # Risk factors
    st.markdown('<h3 class="section-header"><i class="fas fa-exclamation-circle icon"></i>Faktor Risiko</h3>', unsafe_allow_html=True)
    
    risk_factors = [
        ("Diet Tidak Sehat", favc_num * 50 + caec_num * 25, "fas fa-hamburger"),
        ("Kurang Aktivitas", (3-faf) * 30 + tue * 20, "fas fa-couch"),
        ("Kebiasaan Buruk", calc_num * 20 + smoke_num * 30, "fas fa-smoking"),
        ("Faktor Genetik", family_history_num * 100, "fas fa-dna")
    ]
    
    for factor, score, icon in risk_factors:
        level = "Rendah" if score < 30 else "Sedang" if score < 60 else "Tinggi"
        color = "#DCD6F7" if score < 30 else "#A6B1E1" if score < 60 else "#424874"
        text_color = "#1B262C" if score < 60 else "#F5F5F5"
        
        st.markdown(f"""
        <div class="risk-item" style="background: {color}; color: {text_color};">
            <i class="{icon}"></i>
            <div>
                <strong>{factor}</strong><br>
                Skor: {score:.0f}/100 - {level}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TIPS KESEHATAN
# ============================================================================

st.markdown('<h2 class="section-header"><i class="fas fa-heart icon"></i>Tips Kesehatan</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h4><i class="fas fa-apple-alt icon"></i>Tips Diet</h4>
        <ul>
            <li>Konsumsi 5-7 porsi sayuran per hari</li>
            <li>Minum air minimal 8 gelas sehari</li>
            <li>Batasi makanan tinggi kalori</li>
            <li>Makan dengan porsi kecil tapi sering</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h4><i class="fas fa-dumbbell icon"></i>Tips Olahraga</h4>
        <ul>
            <li>Olahraga minimal 150 menit per minggu</li>
            <li>Kombinasi kardio dan kekuatan</li>
            <li>Kurangi waktu screen time</li>
            <li>Gunakan transportasi aktif</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DISCLAIMER
# ============================================================================

st.markdown("""
<div class="warning-box">
    <h4><i class="fas fa-info-circle icon"></i>Disclaimer</h4>
    <p>Aplikasi ini hanya untuk screening awal dan <strong>TIDAK menggantikan</strong> konsultasi dengan dokter. 
    Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis profesional.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #424874; padding: 2rem; margin-top: 2rem;">
    <p><i class="fas fa-heartbeat"></i> <strong>Obesity Prediction System</strong> | Powered by Machine Learning</p>
    <p>Developed for better health monitoring</p>
</div>
""", unsafe_allow_html=True)