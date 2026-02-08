import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    """Load the trained model, scaler, and encoders"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(BASE_DIR, "car_model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        with open(os.path.join(BASE_DIR, "car_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.stop()

# Load the models
model, scaler, encoders = load_models()

# App Header
st.markdown('<p class="main-header">🚗 Ford Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant price estimates for used Ford vehicles based on key features</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app uses machine learning to predict the price of used Ford cars based on various features.
    
    **Features used:**
    - Car Model
    - Year
    - Transmission Type
    - Mileage
    - Fuel Type
    - Road Tax
    - Fuel Efficiency (MPG)
    - Engine Size
    """)
    
    st.header("📊 Model Info")
    st.write(f"**Available Models:** {len(encoders['model'].classes_)}")
    st.write(f"**Fuel Types:** {len(encoders['fuelType'].classes_)}")
    st.write(f"**Transmission Types:** {len(encoders['transmission'].classes_)}")
    
    st.header("💡 Tips")
    st.markdown("""
    - Enter accurate details for best results
    - Lower mileage typically means higher value
    - Newer cars generally have higher prices
    - Fuel efficiency affects pricing
    """)

# Main content area
st.markdown("---")
st.subheader("Enter Car Details")

# Get options from encoders
model_options = sorted(encoders['model'].classes_)
trans_options = sorted(encoders['transmission'].classes_)
fuel_options = sorted(encoders['fuelType'].classes_)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🚙 Vehicle Information")
        car_model = st.selectbox(
            "Car Model",
            model_options,
            help="Select the Ford model"
        )
        year = st.number_input(
            "Year",
            min_value=1990,
            max_value=2025,
            value=2018,
            step=1,
            help="Year of manufacture"
        )
        transmission = st.selectbox(
            "Transmission",
            trans_options,
            help="Type of transmission"
        )
    
    with col2:
        st.markdown("##### ⛽ Fuel & Performance")
        fuelType = st.selectbox(
            "Fuel Type",
            fuel_options,
            help="Type of fuel used"
        )
        mpg = st.number_input(
            "MPG (Miles Per Gallon)",
            min_value=0.0,
            max_value=200.0,
            value=55.0,
            step=0.1,
            help="Fuel efficiency"
        )
        engineSize = st.number_input(
            "Engine Size (L)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="Engine displacement in liters"
        )
    
    with col3:
        st.markdown("##### 📊 Usage & Tax")
        mileage = st.number_input(
            "Mileage",
            min_value=0,
            max_value=500000,
            value=15000,
            step=1000,
            help="Total miles driven"
        )
        tax = st.number_input(
            "Road Tax (£)",
            min_value=0,
            max_value=1000,
            value=150,
            step=10,
            help="Annual road tax"
        )
    
    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

# Prediction logic
if submitted:
    try:
        with st.spinner("🤔 Analyzing car details..."):
            # Encode categorical values
            model_enc = encoders['model'].transform([car_model])[0]
            trans_enc = encoders['transmission'].transform([transmission])[0]
            fuel_enc = encoders['fuelType'].transform([fuelType])[0]
            
            # Create input DataFrame with exact column order
            input_data = pd.DataFrame({
                'model': [model_enc],
                'year': [year],
                'transmission': [trans_enc],
                'mileage': [mileage],
                'fuelType': [fuel_enc],
                'tax': [tax],
                'mpg': [mpg],
                'engineSize': [engineSize]
            })
            
            # Scale the data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Main prediction display
            st.markdown(
                f'<div class="prediction-box">Estimated Price: £{prediction:,.2f}</div>',
                unsafe_allow_html=True
            )
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_range_lower = prediction * 0.90
                price_range_upper = prediction * 1.10
                st.metric(
                    "Price Range (±10%)",
                    f"£{prediction:,.0f}",
                    delta=f"£{price_range_lower:,.0f} - £{price_range_upper:,.0f}"
                )
            
            with col2:
                price_per_mile = prediction / max(mileage, 1)
                st.metric(
                    "Price per Mile",
                    f"£{price_per_mile:.2f}",
                    help="Estimated value per mile driven"
                )
            
            with col3:
                depreciation = max(0, 2025 - year)
                annual_depreciation = prediction / max(depreciation, 1)
                st.metric(
                    "Annual Depreciation",
                    f"£{annual_depreciation:,.0f}",
                    help="Estimated annual value loss"
                )
            
            # Summary box
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"""
            **Summary:**
            - **Model:** {car_model}
            - **Year:** {year} ({2025 - year} years old)
            - **Mileage:** {mileage:,} miles
            - **Fuel:** {fuelType}
            - **Transmission:** {transmission}
            - **Efficiency:** {mpg} MPG
            - **Engine:** {engineSize}L
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download option
            result_df = pd.DataFrame({
                'Feature': ['Model', 'Year', 'Transmission', 'Mileage', 'Fuel Type', 'Tax', 'MPG', 'Engine Size', 'Predicted Price'],
                'Value': [car_model, year, transmission, mileage, fuelType, tax, mpg, engineSize, f"£{prediction:,.2f}"]
            })
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name=f"car_price_prediction_{car_model}_{year}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Prediction completed successfully!")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {str(e)}")
        st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚗 Ford Car Price Predictor | Built with Streamlit & Machine Learning</p>
    <p style='font-size: 0.8rem;'>Predictions are estimates based on historical data and may not reflect current market conditions.</p>
</div>
""", unsafe_allow_html=True)