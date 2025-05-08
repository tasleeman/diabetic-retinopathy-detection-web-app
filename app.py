
import streamlit as st
import torch
from PIL import Image
from model import DRModel
from predict import predict_dr
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sys
from io import BytesIO
import requests

# Configuration
st.set_page_config(
    page_title="DR Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 0,
        'form_data': {},
        'prediction': None,
        'model_loaded': False
    })

# Cache resources
@st.cache_resource
def load_model():
    try:
        model = DRModel()
        model.load_state_dict(
            torch.load('dr_model.pth', map_location=torch.device('cpu'))
        )
        model.eval()
        st.session_state.model_loaded = True
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Helper functions
def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    if height_cm <= 0:
        return 0
    return weight_kg / ((height_cm/100) ** 2)

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    return "Obese"

# Form steps
def step_enter_name():
    st.title("Diabetic Retinopathy Detection")
    with st.form("name_form"):
        st.subheader("Step 1: Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", key="first_name")
        with col2:
            last_name = st.text_input("Last Name", key="last_name")
        
        if st.form_submit_button("Next"):
            if first_name and last_name:
                st.session_state.form_data.update({
                    'first_name': first_name,
                    'last_name': last_name
                })
                st.session_state.current_step += 1
                st.rerun()
            else:
                st.warning("Please enter both names")

def step_enter_gender_age():
    st.title("Diabetic Retinopathy Detection")
    with st.form("gender_age_form"):
        st.subheader("Step 2: Demographic Information")
        gender = st.radio("Gender", ["Male", "Female", "Other"], key="gender")
        age = st.number_input("Age", min_value=1, max_value=120, key="age")
        
        if st.form_submit_button("Next"):
            st.session_state.form_data.update({
                'gender': gender,
                'age': age
            })
            st.session_state.current_step += 1
            st.rerun()

def step_enter_height_weight():
    st.title("Diabetic Retinopathy Detection")
    with st.form("height_weight_form"):
        st.subheader("Step 3: Physical Measurements")
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Height (cm)", min_value=50, max_value=250, key="height")
        with col2:
            weight = st.number_input("Weight (kg)", min_value=10, max_value=300, key="weight")
        
        if st.form_submit_button("Next"):
            st.session_state.form_data.update({
                'height': height,
                'weight': weight
            })
            st.session_state.current_step += 1
            st.rerun()

def step_upload_image():
    st.title("Diabetic Retinopathy Detection")
    st.subheader("Step 4: Retinal Image Analysis")
    
    model = load_model()
    
    tab1, tab2 = st.tabs(["Upload Image", "Paste Image URL"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a retinal image", 
                                       type=['jpg', 'jpeg', 'png'],
                                       accept_multiple_files=False)
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                if st.button("Analyze Image"):
                    with st.spinner('Analyzing...'):
                        st.session_state.prediction = predict_dr(model, image)
                        st.session_state.current_step += 1
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with tab2:
        image_url = st.text_input("Image URL")
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                st.image(image, caption='URL Image', use_column_width=True)
                
                if st.button("Analyze URL Image"):
                    with st.spinner('Analyzing...'):
                        st.session_state.prediction = predict_dr(model, image)
                        st.session_state.current_step += 1
                        st.rerun()
            except Exception as e:
                st.error(f"Couldn't load image: {str(e)}")

def step_results():
    st.title("Detection Results")
    
    # Patient info
    with st.expander("Patient Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.form_data.get('first_name', '')} {st.session_state.form_data.get('last_name', '')}")
            st.write(f"**Age:** {st.session_state.form_data.get('age', '')}")
        with col2:
            st.write(f"**Gender:** {st.session_state.form_data.get('gender', '')}")
            bmi = calculate_bmi(st.session_state.form_data.get('weight', 0), 
                              st.session_state.form_data.get('height', 1))
            st.write(f"**BMI:** {bmi:.1f} ({get_bmi_category(bmi)})")
    
    # Results
    if st.session_state.prediction:
        st.subheader("Prediction Results")
        df = pd.DataFrame({
            'Condition': list(st.session_state.prediction.keys()),
            'Probability': list(st.session_state.prediction.values())
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(df, x='Condition', y='Probability',
                    color='Condition', text=df['Probability'].apply(lambda x: f"{x:.1%}"),
                    template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        max_condition = max(st.session_state.prediction, key=st.session_state.prediction.get)
        recommendations = {
            'No DR': "✅ No diabetic retinopathy detected. Continue annual eye exams.",
            'Mild DR': "⚠️ Mild DR detected. Schedule follow-up in 6-12 months.",
            'Moderate DR': "⚠️ Moderate DR detected. Schedule follow-up in 3-6 months.",
            'Severe DR': "❗ Severe DR detected. Consult ophthalmologist within 1 month.",
            'Proliferative DR': "❗❗ Proliferative DR detected. Immediate consultation required."
        }
        st.subheader("Recommendation")
        st.markdown(f"**{max_condition}**: {recommendations.get(max_condition, '')}")
    
    if st.button("New Analysis"):
        st.session_state.current_step = 0
        st.rerun()

# Main app flow
def main():
    steps = {
        0: step_enter_name,
        1: step_enter_gender_age,
        2: step_enter_height_weight,
        3: step_upload_image,
        4: step_results
    }
    steps[st.session_state.current_step]()

if __name__ == "__main__":
    main()