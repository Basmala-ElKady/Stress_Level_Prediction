import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="💓",
    layout="wide"
)

# Modern Clean Theme with Google Fonts and Font Awesome
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
<style>
    /* Import Inter font for modern look */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: #1a202c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #2d3748;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        color: #2d3748;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        border: 2px solid #22c55e;
        margin: 2rem 0;
        color: #166534;
        box-shadow: 0 4px 6px rgba(34, 197, 94, 0.1);
    }
    
    .tips-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #f59e0b;
        margin: 1rem 0;
        color: #92400e;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    
    .encouragement-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #3b82f6;
        margin: 1rem 0;
        color: #1e40af;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    .input-section {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        color: #475569;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Form labels */
    .stNumberInput label {
        font-weight: 500;
        color: #374151;
        font-family: 'Inter', sans-serif;
    }
    
    /* Help text */
    .stTooltip {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics styling */
    .stMetric > label {
        color: #6b7280;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric > div > h3 {
        color: #111827;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Icons styling */
    .fa {
        margin-right: 0.5rem;
        color: #6b7280;
    }
    
    /* List styling */
    .tips-card ul, .encouragement-card ul {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    .tips-card li, .encouragement-card li {
        margin: 0.5rem 0;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Footer styling */
    .footer-note {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scalers
@st.cache_resource
def load_model_and_scalers():
    try:
        # Use relative paths from the app directory
        model = keras.models.load_model('../models/stress_nn_model.h5')
        scaler_X = joblib.load('../models/scaler_X.pkl')
        scaler_y = joblib.load('../models/scaler_y.pkl')
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">💓 Stress Level Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Enter your heart rate data to predict stress levels")
    
    # Load model
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    if model is None:
        st.error("Could not load the prediction model. Please check if the model files exist.")
        return
    
    # Create sidebar for information
    with st.sidebar:
        st.markdown("### ℹ️ About This App")
        
        # How it works section
        with st.expander("🔍 How it works", expanded=False):
            st.markdown("""
            This app uses a neural network trained on heart rate variability data to predict stress levels. 
            The model analyzes four key features of your heart rate data to make predictions.
            """)
        
        # Features explanation
        with st.expander("📊 Features Explained", expanded=False):
            st.markdown("""
            **🕐 Average Heartbeat Interval:** Average time between heartbeats in milliseconds
            
            **📊 Heart Rate Variability:** How much your heart rate varies between beats
            
            **🔴 Activity Level:** Low frequency component indicating stress response
            
            **🟢 Relaxation Level:** High frequency component indicating relaxation state
            """)
        
        # Tips for better results
        with st.expander("💡 Tips for Better Results", expanded=False):
            st.markdown("""
            • Enter data from a calm, resting state
            • Use data from multiple measurements if possible
            • Ensure you're not under immediate stress when measuring
            • Consider measuring at different times of day
            """)
        
        # About the model
        with st.expander("🤖 About the Model", expanded=False):
            st.markdown("""
            • **Type:** Neural Network (Deep Learning)
            • **Training:** 10,000+ heart rate samples
            • **Features:** 4 heart rate variability metrics
            • **Output:** Predicted heart rate and stress level
            """)
    
    # Main content area
    st.markdown('<h3 class="sub-header">📊 Enter Your Heart Rate Data</h3>', unsafe_allow_html=True)
    
    # Input form
    with st.form("stress_prediction_form"):
        st.markdown("### 💓 Heart Rate Features")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            mean_rr = st.number_input(
                "🕐 Average Heartbeat Interval (ms):", 
                min_value=400.0, 
                max_value=1200.0, 
                value=800.0,
                step=10.0,
                help="Average time between heartbeats in milliseconds"
            )
            
            sd_rr = st.number_input(
                "📊 Heart Rate Variability (ms):", 
                min_value=10.0, 
                max_value=200.0, 
                value=100.0,
                step=5.0,
                help="Variability in heart rate intervals"
            )
        
        with col2:
            lf = st.number_input(
                "🔴 Activity Level:", 
                min_value=100.0, 
                max_value=3000.0, 
                value=1000.0,
                step=50.0,
                help="Low frequency component of heart rate variability"
            )
            
            hf = st.number_input(
                "🟢 Relaxation Level:", 
                min_value=5.0, 
                max_value=500.0, 
                value=50.0,
                step=5.0,
                help="High frequency component of heart rate variability"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict My Stress Level", use_container_width=True)
    
    # Prediction logic
    if submitted:
        try:
            # Prepare input data
            input_data = np.array([[mean_rr, sd_rr, lf, hf]])
            
            # Scale the input
            input_scaled = scaler_X.transform(input_data)
            
            # Make prediction
            prediction_scaled = model.predict(input_scaled, verbose=0)
            prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
            
            # Display results
            st.markdown("---")
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Create result display
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            
            with col_result2:
                st.markdown(f"""
                <div class="prediction-result">
                <h2>Predicted Heart Rate: {prediction:.1f} BPM</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">
                Based on your heart rate variability data, the model predicts your heart rate would be approximately <strong>{prediction:.1f} beats per minute</strong>.
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation with tips and encouragement
            st.markdown("### 📈 Your Stress Level Analysis")
            
            if prediction < 60:
                stress_level = "Low"
                color = "🟢"
                interpretation = "Your predicted heart rate suggests a relaxed state with low stress levels."
                tips = """
                <div class="encouragement-card">
                <h3><i class="fas fa-star"></i> Excellent! You're doing great!</h3>
                <p><strong>Keep up the good work:</strong></p>
                <ul>
                <li><i class="fas fa-check-circle"></i> Continue your current lifestyle habits</li>
                <li><i class="fas fa-dumbbell"></i> Maintain regular exercise routine</li>
                <li><i class="fas fa-leaf"></i> Keep practicing stress management techniques</li>
                <li><i class="fas fa-bed"></i> Get adequate sleep (7-9 hours)</li>
                <li><i class="fas fa-apple-alt"></i> Stay hydrated and eat balanced meals</li>
                </ul>
                <p><em>You're managing stress very well! Keep it up! <i class="fas fa-thumbs-up"></i></em></p>
                </div>
                """
            elif prediction < 80:
                stress_level = "Moderate"
                color = "🟡"
                interpretation = "Your predicted heart rate indicates moderate stress levels, which is normal for daily activities."
                tips = """
                <div class="tips-card">
                <h3><i class="fas fa-balance-scale"></i> Moderate Stress - You're doing okay!</h3>
                <p><strong>Some helpful tips to maintain balance:</strong></p>
                <ul>
                <li><i class="fas fa-lungs"></i> Practice 10 minutes of deep breathing daily</li>
                <li><i class="fas fa-walking"></i> Take short walks during breaks</li>
                <li><i class="fas fa-tint"></i> Drink more water throughout the day</li>
                <li><i class="fas fa-moon"></i> Ensure 7-8 hours of quality sleep</li>
                </ul>
                <p><em>You're handling stress well! Small improvements can make a big difference! <i class="fas fa-star"></i></em></p>
                </div>
                """
            elif prediction < 100:
                stress_level = "High"
                color = "🟠"
                interpretation = "Your predicted heart rate suggests elevated stress levels. Consider relaxation techniques."
                tips = """
                <div class="tips-card">
                <h3><i class="fas fa-exclamation-triangle"></i> High Stress - Time to take action!</h3>
                <p><strong>Important steps to reduce stress:</strong></p>
                <ul>
                <li><i class="fas fa-om"></i> Practice meditation or mindfulness for 15-20 minutes daily</li>
                <li><i class="fas fa-running"></i> Engage in regular physical exercise (30 minutes, 3x/week)</li>
                <li><i class="fas fa-spa"></i> Try progressive muscle relaxation techniques</li>
                <li><i class="fas fa-mobile-alt"></i> Take digital breaks and limit screen time</li>
                <li><i class="fas fa-apple-alt"></i> Eat nutritious meals and avoid excessive caffeine</li>
                <li><i class="fas fa-bed"></i> Prioritize sleep - aim for 8 hours nightly</li>
                <li><i class="fas fa-users"></i> Talk to friends, family, or a counselor</li>
                </ul>
                <p><em>Remember: It's okay to ask for help! You've got this! <i class="fas fa-heart"></i></em></p>
                </div>
                """
            else:
                stress_level = "Very High"
                color = "🔴"
                interpretation = "Your predicted heart rate indicates very high stress levels. Please consult a healthcare professional."
                tips = """
                <div class="tips-card">
                <h3><i class="fas fa-exclamation-circle"></i> Very High Stress - Please seek support!</h3>
                <p><strong>Immediate actions to take:</strong></p>
                <ul>
                <li><i class="fas fa-hospital"></i> <strong>Consult a healthcare professional immediately</strong></li>
                <li><i class="fas fa-lungs"></i> Practice deep breathing exercises right now</li>
                <li><i class="fas fa-tree"></i> Take a gentle walk in nature</li>
                <li><i class="fas fa-phone"></i> Reach out to a trusted friend or family member</li>
                <li><i class="fas fa-bed"></i> Ensure you're getting adequate rest</li>
                <li><i class="fas fa-apple-alt"></i> Focus on proper nutrition and hydration</li>
                <li><i class="fas fa-mobile-alt"></i> Consider a digital detox</li>
                </ul>
                <p><em><strong>Your health is important! Don't hesitate to seek professional help! <i class="fas fa-user-md"></i></strong></em></p>
                </div>
                """
            
            col_int1, col_int2, col_int3 = st.columns([1, 2, 1])
            with col_int2:
                st.markdown(f"""
                <div class="metric-card">
                <h3>{color} Stress Level: {stress_level}</h3>
                <p style="font-size: 1.1rem;">{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display tips and encouragement
            st.markdown(tips, unsafe_allow_html=True)
            
            # Input summary
            st.markdown("### 📋 Your Input Summary")
            input_df = pd.DataFrame({
                'Feature': ['Mean RR Interval', 'RR Standard Deviation', 'Low Frequency Power', 'High Frequency Power'],
                'Value': [f"{mean_rr:.1f} ms", f"{sd_rr:.1f} ms", f"{lf:.1f}", f"{hf:.1f}"]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-note" style="text-align: center; margin-top: 2rem; padding: 2rem;">
    <h4 style="margin-bottom: 1rem;"><i class="fas fa-info-circle"></i> Important Note</h4>
    <p style="font-size: 1rem; line-height: 1.6;">
    This is a demonstration application for educational purposes. The predictions are based on machine learning models and should not replace professional medical advice. 
    <br><br>
    <strong>For any health concerns or medical decisions, please consult a qualified healthcare professional.</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()