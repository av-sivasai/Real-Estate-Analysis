import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Real Estate Predictor", page_icon="🏠", layout="wide")

# 2. Load the Saved Models
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('models/price_model.pkl')
        clf_model = joblib.load('models/class_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return reg_model, clf_model, scaler
    except FileNotFoundError:
        return None, None, None

reg_model, clf_model, scaler = load_models()

# 3. Header
st.title("🏡 AI Real Estate Consultant")
st.markdown("### Adjust the details below to get a price estimate:")

if reg_model is None:
    st.error("⚠️ Models not found! Check your 'models' folder.")
    st.stop()

# 4. User Inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📍 **Space & Size**")
    # FIX: We set min_value=0 so you CAN type 10, but the logic below will catch it.
    GrLivArea = st.number_input("Living Area (Sq Ft)", min_value=0, max_value=10000, value=1500, step=50)
    BedroomAbvGr = st.selectbox("🛏️ Bedrooms", [1, 2, 3, 4, 5, 6], index=2)

with col2:
    st.info("🛠️ **Condition & Quality**")
    OverallQual = st.slider("Overall Quality (1=Poor, 10=Luxury)", 1, 10, 5)
    YearBuilt = st.number_input("📅 Year Built", min_value=1900, max_value=2025, value=2000)

with col3:
    st.info("🛁 **Amenities**")
    FullBath = st.radio("Bathrooms", [1, 2, 3, 4], horizontal=True)

# 5. Validation Logic
# This runs every time you change a number
validation_error = None  # Start with no error

if GrLivArea < 500:
    validation_error = "⚠️ Error: Living Area is too small! It must be at least 500 Sq Ft to estimate."

if BedroomAbvGr > 4 and GrLivArea < 1000:
    validation_error = "⚠️ Error: Unrealistic dimensions! A house with 5+ bedrooms needs more than 1000 Sq Ft."

st.divider()

# 6. Prediction Section
if st.button("🚀 Estimate Price & Value", type="primary", use_container_width=True):
    
    if validation_error:
        # IF there is an error, show it and STOP. Do not predict.
        st.error(validation_error)
    else:
        # ELSE (no error), run the prediction.
        
        # Prepare data
        input_data = {
            'OverallQual': OverallQual,
            'YearBuilt': YearBuilt,
            'GrLivArea': GrLivArea,
            'FullBath': FullBath,
            'BedroomAbvGr': BedroomAbvGr
        }
        input_df = pd.DataFrame(input_data, index=[0])

        # Scale and Predict
        input_scaled = scaler.transform(input_df)
        price_pred = reg_model.predict(input_scaled)[0]
        class_pred = clf_model.predict(input_scaled)[0]
        
        neighborhood_status = "🌟 Premium / High-Value" if class_pred == 1 else "🏘️ Standard / Affordable"

        # Display Results
        st.balloons()
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.success("### 💰 Estimated Market Value")
            st.metric(label="Price", value=f"${price_pred:,.2f}")
            
        with res_col2:
            st.info("### 🏙️ Neighborhood Rating")

            st.write(f"Based on the specs, this property fits into a **{neighborhood_status}** category.")
