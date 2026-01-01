import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests
import numpy as np 
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="PSafe",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main theme - Soft sage green background */
    .stApp {
        background: linear-gradient(135deg, #ECECEA 0%, #e8ecf0 100%);
    }
    
    .main {
        padding: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-weight: 500;
        background: white;
        color: #153351;
        border: 1.5px solid #d4e4e1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stButton>button:hover {
        background: #f7fbfa;
        border-color: #8fa3b8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(45, 95, 93, 0.1);
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #153351 0%, #153351 100%);
        color: white;
        border: none;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #153351 0%, #153351 100%);
        box-shadow: 0 6px 20px rgba(90, 158, 152, 0.3);
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        border-radius: 10px;
        background: white;
        color: #153351;
        border: 1.5px solid #e0ebe8;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #153351;
        box-shadow: 0 0 0 3px rgba(90, 158, 152, 0.1);
        background: #fafcfb;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #a8bac5;
    }
    
    /* Info card header */
    .info-card {
        background: linear-gradient(135deg, #153351 0%, #153351 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(90, 158, 152, 0.2);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #e8ecf0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(90, 158, 152, 0.15);
    }
    
    /* Headers */
    h1 {
        color: #153351;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #153351;
        font-weight: 600;
    }
    
    h3 {
        color: #153351;
        font-weight: 500;
    }
    
    /* Prediction section */
    .prediction-section {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-top: 2rem;
        border: 1px solid #e8ecf0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    }
    
    .prediction-section h3 {
        color: #153351!important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #153351;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e8ecf0;
    }
    
    /* Labels */
    .stTextInput>label {
        color: #1533510 !important;
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Alert boxes with calm styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5f3 0%, #c1d1db 100%);
        border-left: 4px solid #153351;
        color: #153351;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, #fff8e8 0%, #ffefd4 100%);
        border-left: 4px solid #f0b849;
        color: #8b6914;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #fdeef0 0%, #fce4e7 100%);
        border-left: 4px solid #e57373;
        color: #8b3a3a;
    }
    
    /* Info message */
    .stInfo {
        background: linear-gradient(135deg, #e8f0f8 0%, #d4e8f0 100%);
        border-left: 4px solid #5a9eb8;
        color: #29415c;
    }
    
    /* Markdown text */
    p {
        color: #23415e;
        line-height: 1.7;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #ECECEA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #8fa3b8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #153351;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
maternal_model = joblib.load("model/maternal_model.pkl")
fetal_model = joblib.load("model/fetal_model.pkl")

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white; font-weight: 600; margin-bottom: 0.5rem;'>PSafe</h1>", unsafe_allow_html=True)

    selected = option_menu(
        'Navigation',
        ['About us', 'Pregnancy Risk Prediction', 'Fetal Health Prediction', 'Dashboard'],
        icons=['info-circle', 'heart-pulse', 'activity', 'graph-up'],
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "transparent"},
            "icon": {"color": "#d0ddec", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "8px 0",
                "padding": "12px 16px",
                "color": "#d4dde4",
                "background-color": "rgba(255, 255, 255, 0.05)",
                "--hover-color": "rgba(255, 255, 255, 0.1)",
                "border-radius": "10px",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 255, 255, 0.15)",
                "color": "white",
                "font-weight": "500"
            },
        }
    )
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 11px; color: #a8bac5; margin-top: 2rem;'>© 2025 PSafe<br>All Rights Reserved</p>", unsafe_allow_html=True)

# About Us Page
if selected == 'About us':
    st.markdown("<div class='info-card'><h1 style='color: white; text-align: center; margin-bottom: 1rem;'>Welcome to PSafe</h1><p style='text-align: center; font-size: 18px; color: rgba(255,255,255,0.95); font-weight: 300;'>Revolutionizing Healthcare Through Predictive Analysis</p></div>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 16px; line-height: 1.8; color: #23415e; text-align: center; max-width: 800px; margin: 0 auto 3rem;'>At PSafe, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate predictions and proactive risk management.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<h3 style='color: #153351;'>Pregnancy Risk Prediction</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #1c2e4a; line-height: 1.7;'>Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of potential risks during pregnancy.</p>", unsafe_allow_html=True)
        st.image("graphics/pregnancy_risk_image.png")

    
    with col2:
        
        st.markdown("<h3 style='color: #153351;'>Fetal Health Prediction</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #1c2e4a; line-height: 1.7;'>Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, and genetic factors, we deliver insights into the well-being of the unborn child.</p>", unsafe_allow_html=True)
        st.image("graphics/fetal_health_image.png")
      

   
    st.markdown("<h3 style='color: #153351;'>Dashboard</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #1c2e4a; line-height: 1.7;'>Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard is designed for ease of use and accessibility.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Thank you for choosing PSafe. We are committed to advancing healthcare through technology and predictive analytics. Feel free to explore our features and take advantage of the insights we provide.")

# Pregnancy Risk Prediction Page
if selected == 'Pregnancy Risk Prediction':
    st.markdown("<div class='info-card'><h1 style='color: white; font-weight: 600;'>Pregnancy Risk Prediction</h1></div>", unsafe_allow_html=True)
    
    st.info("Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health.")

    st.markdown("<h3 style='color: #153351;'>Enter Patient Information</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #1c2e4a; margin-bottom: 1.5rem;'>Please fill in all required fields for accurate prediction</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        age = st.text_input('Age of the Person', key="age", placeholder="Enter age")
        bodyTemp = st.text_input('Body Temperature (°C)', placeholder="Enter temperature")
        
    with col2:
        diastolicBP = st.text_input('Diastolic BP (mmHg)', placeholder="Enter BP")
        heartRate = st.text_input('Heart Rate (bpm)', placeholder="Enter heart rate")
    
    with col3:
        BS = st.text_input('Blood Glucose (mmol/L)', placeholder="Enter glucose level")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    scale_X = joblib.load("model/scaler.pkl")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_btn = st.button('Predict Risk', type="primary")
    with col2:
        clear_btn = st.button("Clear All", type="secondary")
    
    if predict_btn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_data = np.array([[age, diastolicBP, BS, bodyTemp, heartRate]])
            input_scaled = scale_X.transform(input_data)
            predicted_risk = maternal_model.predict(input_scaled)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #153351;'>Prediction Result</h3>", unsafe_allow_html=True)
        
        if predicted_risk[0] == 0:
            st.success("Low Risk — The pregnancy shows minimal risk factors. Continue regular monitoring.")
        elif predicted_risk[0] == 1:
            st.warning("Medium Risk — Moderate risk factors detected. Enhanced monitoring recommended.")
        else:
            st.error("High Risk — Significant risk factors present. Immediate medical attention advised.")
    
    if clear_btn:
        st.rerun()

# Fetal Health Prediction Page
if selected == 'Fetal Health Prediction':
    st.markdown("<div class='info-card'><h1 style='color: white; font-weight: 600;'>Fetal Health Prediction</h1></div>", unsafe_allow_html=True)
    
    st.info("Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality.")
    
   
    st.markdown("<h3 style='color: #153351;'>Enter CTG Parameters</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #1c2e4a; margin-bottom: 1.5rem;'>Please provide all cardiotocogram measurements</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        BaselineValue = st.text_input('Baseline Value', placeholder="Enter value")
        uterine_contractions = st.text_input('Uterine Contractions', placeholder="Enter value")
        prolongued_decelerations = st.text_input('Prolongued Decelerations', placeholder="Enter value")
        percentage_of_time_with_abnormal_long_term_variability = st.text_input('Percentage of Time With ALTV', placeholder="Enter value")
        histogram_min = st.text_input('Histogram Min', placeholder="Enter value")
        histogram_number_of_zeroes = st.text_input('Histogram Number of Zeroes', placeholder="Enter value")
        histogram_median = st.text_input('Histogram Median', placeholder="Enter value")
        
    with col2:
        Accelerations = st.text_input('Accelerations', placeholder="Enter value")
        light_decelerations = st.text_input('Light Decelerations', placeholder="Enter value")
        abnormal_short_term_variability = st.text_input('Abnormal Short Term Variability', placeholder="Enter value")
        mean_value_of_long_term_variability = st.text_input('Mean Value Long Term Variability', placeholder="Enter value")
        histogram_max = st.text_input('Histogram Max', placeholder="Enter value")
        histogram_mode = st.text_input('Histogram Mode', placeholder="Enter value")
        histogram_variance = st.text_input('Histogram Variance', placeholder="Enter value")
    
    with col3:
        fetal_movement = st.text_input('Fetal Movement', placeholder="Enter value")
        severe_decelerations = st.text_input('Severe Decelerations', placeholder="Enter value")
        mean_value_of_short_term_variability = st.text_input('Mean Value Short Term Variability', placeholder="Enter value")
        histogram_width = st.text_input('Histogram Width', placeholder="Enter value")
        histogram_number_of_peaks = st.text_input('Histogram Number of Peaks', placeholder="Enter value")
        histogram_mean = st.text_input('Histogram Mean', placeholder="Enter value")
        histogram_tendency = st.text_input('Histogram Tendency', placeholder="Enter value")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_btn = st.button('Predict Fetal Health', type="primary")
    with col2:
        clear_btn = st.button("Clear All", type="secondary")
    
    if predict_btn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_risk = fetal_model.predict([[
                BaselineValue, Accelerations, fetal_movement, uterine_contractions,
                light_decelerations, severe_decelerations, prolongued_decelerations,
                abnormal_short_term_variability, mean_value_of_short_term_variability,
                percentage_of_time_with_abnormal_long_term_variability,
                mean_value_of_long_term_variability, histogram_width, histogram_min,
                histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes,
                histogram_mode, histogram_mean, histogram_median, histogram_variance,
                histogram_tendency
            ]])
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Prediction Result")
        
        if predicted_risk[0] == 0:
            st.success("Normal — Fetal health parameters are within normal range.")
        elif predicted_risk[0] == 1:
            st.warning("Suspect — Some abnormal indicators detected. Further monitoring recommended.")
        else:
            st.error("Pathological — Critical indicators present. Immediate medical intervention required.")
    
    if clear_btn:
        st.rerun()





# Dashboard Page
if selected == 'Dashboard':
    st.markdown("<div class='info-card'><h1>Interactive Dashboard</h1></div>", unsafe_allow_html=True)
    st.info("This dashboard shows live patient predictions. Use filters to explore the data.")

    API_URL = "https://mocki.io/v1/b51b027d-fe68-455a-93bf-590e24f7d1df" 

    @st.cache_data
    def load_data():
        response = requests.get(API_URL)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    try:
        df = load_data()
    except Exception as e:
        st.error("Unable to load data: " + str(e))
        st.stop()

    # Sidebar Filters
    st.sidebar.header("Filters")
    age_range = st.sidebar.slider("Age Range", int(df.age.min()), int(df.age.max()), (20, 40))
    risk_filter = st.sidebar.multiselect("Risk Level", options=df.risk.unique(), default=df.risk.unique())
    df['date'] = pd.to_datetime(df['date'])
    date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])
    filtered_df = df[
        (df.age.between(age_range[0], age_range[1])) &
        (df.risk.isin(risk_filter)) &
        (df.date.between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]

    #KPIs
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div style='color:#153351; font-size: 1.2rem; font-weight: 600;'>Total Patients<br>{len(filtered_df)}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='color:#153351; font-size: 1.2rem; font-weight: 600;'>High Risk (%)<br>{(filtered_df.risk=='High').mean()*100:.1f}%</div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='color:#153351; font-size: 1.2rem; font-weight: 600;'>Average Age<br>{filtered_df.age.mean():.1f}</div>", unsafe_allow_html=True)
    col4.markdown(f"<div style='color:#153351; font-size: 1.2rem; font-weight: 600;'>Avg Blood Sugar<br>{filtered_df.bs.mean():.1f}</div>", unsafe_allow_html=True)

    #Interactive Charts
    fig1 = px.pie(filtered_df, names='risk', title="Risk Distribution", hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered_df, x='age', y='bs', color='risk', size='diastolic_bp',
                      hover_data=['age','bs','diastolic_bp'], title="Age vs Blood Sugar vs BP")
    st.plotly_chart(fig2, use_container_width=True)

    trend = filtered_df.groupby(filtered_df['date'].dt.to_period("M")).size().reset_index(name="count")
    trend['date'] = trend['date'].dt.to_timestamp()
    fig3 = px.line(trend, x='date', y='count', title="Monthly Predictions Trend", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

    #Data Table + Download
    st.markdown("<h3 style='color:#153351;'>Patient Data Table</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df)
    st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name='patient_data.csv', mime='text/csv')

    # Alert if too many high-risk patients
    high_risk_count = (filtered_df.risk=='High').sum()
    if high_risk_count > 2:
        st.warning(f"⚠️ There are {high_risk_count} high-risk patients in the selected data!")
