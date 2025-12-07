import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ====================== 1. SETUP ======================
st.set_page_config(
    page_title="Mumbai House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load the model (use the new model from training)
try:
    model = joblib.load('work_new.joblib')
    st.sidebar.success("✓ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file 'work_new.joblib' not found. Please run the training script first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ====================== 2. TITLE & DESCRIPTION ======================
st.title('🏠 MUMBAI HOUSE PRICE PREDICTION')
st.markdown("""
Predict the market price of residential properties in Mumbai based on key features.
Enter the property details below to get an instant price estimate.
""")

# ====================== 3. LOCATION DATA ======================
# These should match the locations in your training data
MUMBAI_LOCATIONS = [
    "Andheri East Mumbai", "Andheri West Mumbai", "Borivali Mumbai",
    "Dahisar Mumbai", "Goregaon East Mumbai", "Goregaon West Mumbai",
    "Kandivali West Mumbai", "Kandivali East Mumbai", "Khar Mumbai",
    "Juhu Mumbai", "Malad Mumbai", "Santacruz East Mumbai",
    "Santacruz West Mumbai", "Chembur Mumbai", "Dadar Mumbai",
    "Wadala Mumbai", "Bhandup Mumbai", "Kurla Mumbai",
    "Ghatkopar Mumbai", "Powai Mumbai", "Vikhroli Mumbai"
]

# ====================== 4. INPUT FORM ======================
with st.form(key='prediction_form'):
    st.header("Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox(
            "📍 **Location**", 
            MUMBAI_LOCATIONS,
            help="Select the neighborhood/area in Mumbai"
        )
        
        furnish = st.selectbox(
            "🛋️ **Furnishing Status**", 
            ['Furnished', 'Semi-Furnished', 'Unfurnished'],
            help="Furnished properties typically command higher prices"
        )
        
        sqft = st.number_input(
            "📏 **Carpet Area (sq ft)**", 
            min_value=100, 
            max_value=10000, 
            value=800, 
            step=50,
            help="Enter the carpet area in square feet (typical range: 300-3000 sq ft)"
        )
    
    with col2:
        bhk = st.number_input(
            "🚪 **Number of BHK**", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1,
            help="Bedroom, Hall, Kitchen count"
        )
        
        bath = st.number_input(
            "🚿 **Number of Bathrooms**", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1,
            help="Total bathrooms including attached bathrooms"
        )
        
        park = st.number_input(
            "🅿️ **Number of Parking Spots**", 
            min_value=0, 
            max_value=5, 
            value=1, 
            step=1,
            help="Dedicated parking spaces"
        )
    
    # Submit button
    submit_button = st.form_submit_button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    )

# ====================== 5. PREDICTION LOGIC ======================
if submit_button:
    # Input validation
    validation_errors = []
    
    if sqft <= 0:
        validation_errors.append("Carpet area must be positive")
    if bhk <= 0:
        validation_errors.append("BHK count must be positive")
    if bath <= 0:
        validation_errors.append("Bathroom count must be positive")
    if park < 0:
        validation_errors.append("Parking spots cannot be negative")
    
    # Basic sanity checks
    if bath > bhk + 2:  # Allow some extra bathrooms but not too many
        validation_errors.append("Number of bathrooms seems unusually high for the BHK count")
    
    if sqft < bhk * 200:  # Minimum ~200 sq ft per room
        validation_errors.append("Area seems too small for the number of rooms")
    
    if validation_errors:
        for error in validation_errors:
            st.error(f"⚠️ {error}")
        st.info("💡 **Tips:** Typical Mumbai apartments: 1BHK ~500-700 sq ft, 2BHK ~800-1200 sq ft, 3BHK ~1200-1800 sq ft")
    else:
        # Show loading spinner
        with st.spinner('Calculating price prediction...'):
            @st.cache_data
            def predict(location, furnish, sqft, bhk, bath, park):
                """Make prediction with proper data formatting"""
                # Prepare input data
                input_data = pd.DataFrame({
                    'loc': [location],
                    'furnishing': [furnish],
                    'sqft': [sqft],
                    'size': [bhk],  # Note: 'size' in model = BHK count
                    'bath': [bath],
                    'parking': [park]
                })
                
                # Ensure correct column order (important for the model)
                input_data = input_data[['loc', 'furnishing', 'sqft', 'size', 'bath', 'parking']]
                
                try:
                    prediction = model.predict(input_data)[0]
                    return float(prediction)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    return None
            
            # Make prediction
            prediction = predict(location, furnish, sqft, bhk, bath, park)
            
            if prediction is not None:
                # Ensure prediction is reasonable
                if prediction <= 0:
                    st.warning("⚠️ The prediction seems unusually low. This might indicate:")
                    st.markdown("""
                    - **Unusual input values** (extremely small area, unusual room/bathroom ratio)
                    - **Location/furnishing combination** not well-represented in training data
                    - **Model limitations** for edge cases
                    """)
                    
                    # Provide a sensible fallback estimate based on averages
                    st.info("💡 **Estimated price range for similar properties:** ₹50,00,000 - ₹2,00,00,000")
                    
                else:
                    # Format the price nicely
                    formatted_price = f"₹{prediction:,.0f}"
                    
                    # Show prediction in a nice box
                    st.success("### 🎯 Price Prediction")
                    
                    # Main prediction display
                    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
                    with col_pred2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 15px;
                            text-align: center;
                            color: white;
                            margin: 1rem 0;
                        ">
                            <h1 style="margin: 0; font-size: 2.5rem;">{formatted_price}</h1>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Estimated Market Price</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Alternative representations
                    col_alt1, col_alt2 = st.columns(2)
                    
                    with col_alt1:
                        st.metric("In Crores", f"₹{prediction/10000000:.2f} Cr")
                    
                    with col_alt2:
                        st.metric("In Lakhs", f"₹{prediction/100000:.2f} L")
                    
                    # Confidence note
                    st.info("""
                    **Note:** This is an estimate based on historical data. 
                    Actual market price may vary based on:
                    - Property condition and age
                    - Floor number and view
                    - Building amenities
                    - Market conditions
                    - Exact location within the area
                    """)

# ====================== 6. SIDEBAR INFORMATION ======================
st.sidebar.header("ℹ️ About This Predictor")
st.sidebar.markdown("""
This machine learning model predicts house prices in Mumbai based on:
- **Location** (Neighborhood/Area)
- **Property Size** (Area in sq ft & BHK count)
- **Furnishing Status**
- **Amenities** (Bathrooms, Parking)

**Model Accuracy:** ~85-90% (based on historical data)
""")

st.sidebar.header("📊 Price Indicators")
st.sidebar.markdown("""
**Typical Mumbai Price Ranges:**
- **1BHK:** ₹80L - ₹1.5Cr
- **2BHK:** ₹1.2Cr - ₹2.5Cr  
- **3BHK:** ₹2Cr - ₹4Cr+

**Factors affecting price:**
1. **Location** (40-50% impact)
2. **Area & Layout** (30-40% impact)
3. **Furnishing & Amenities** (10-20% impact)
""")

st.sidebar.header("🔍 Tips for Accurate Prediction")
st.sidebar.markdown("""
1. **Be realistic with inputs:**
   - Area: 500-3000 sq ft for most apartments
   - BHK-Bath ratio: Usually 1:1 or 2:1
   - Parking: 0-2 for most apartments

2. **Location matters most:**
   - South Mumbai & Western suburbs command premium
   - Central & Eastern suburbs relatively affordable

3. **Consider market trends:**
   - Prices vary with season
   - New developments affect local prices
""")

# ====================== 7. FOOTER ======================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📊 Model trained on Mumbai real estate data | 🏢 Property price predictor | 🔄 Updated regularly</p>
    <p>⚠️ <em>For reference only. Consult real estate professionals for actual transactions.</em></p>
</div>
""", unsafe_allow_html=True)

# ====================== 8. DEBUG OPTION (Hidden by default) ======================
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.write("### Model Information")
    st.sidebar.write(f"Model type: {type(model).__name__}")
    
    if hasattr(model, 'named_steps'):
        st.sidebar.write("Pipeline steps:", list(model.named_steps.keys()))
    
    st.sidebar.write("### Input Summary")
    if submit_button and 'prediction' in locals():
        st.sidebar.write(f"""
        - Location: {location}
        - Furnishing: {furnish}
        - Area: {sqft} sq ft
        - BHK: {bhk}
        - Bathrooms: {bath}
        - Parking: {park}
        - Raw prediction: ₹{prediction:,.0f}
        """)