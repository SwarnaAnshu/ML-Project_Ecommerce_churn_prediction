"""
E-Commerce Customer Churn Prediction — Deployment App
Run with: streamlit run app.py

Requires: pip install fpdf2 --break-system-packages
"""

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import io
from datetime import datetime
from fpdf import FPDF

st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="📊", layout="centered")

# Hide the per-field "x" clear icon that appears on empty-capable selectboxes.
# Fields can still be reset via the "Clear Form" button.
st.markdown(
    """
    <style>
    [aria-label="Clear value"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Load saved pipeline artifacts ----
model = xgb.XGBClassifier()
model.load_model("trained_model.json")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
imputer = joblib.load("imputer.pkl")
selected_features = joblib.load("selected_features.pkl")

numerical_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                   'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear',
                   'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                     'PreferedOrderCat', 'MaritalStatus']

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---- Initial values shown the first time the app loads ----
# Numeric fields start at 0 so you can just use +/- instead of typing.
NUMERIC_KEYS = [
    "k_tenure", "k_warehouse", "k_delivery_time", "k_order_hike", "k_hours_on_app",
    "k_devices", "k_addresses", "k_coupon_used", "k_order_count",
    "k_days_since_order", "k_cashback"
]
SELECT_KEYS = [
    "k_complain", "k_city_tier", "k_login_device", "k_payment_mode",
    "k_gender", "k_order_cat", "k_marital_status"
]

# Extra e-commerce details — shown in the report/history for context only.
# The trained model doesn't use these, so they're optional (not required to predict).
#
# TODO (migrate once retrained): Once you add NumberOfReturns and SubscriptionStatus
# to your dataset, rerun EDA/feature selection, and retrain trained_model.json + refit
# imputer.pkl/encoder.pkl/scaler.pkl/selected_features.pkl to include them — then move
# the relevant keys below out of OPTIONAL_NUMERIC_KEYS / OPTIONAL_SELECT_KEYS into
# NUMERIC_KEYS / SELECT_KEYS above, and add their real column names to numerical_cols /
# categorical_cols near the top of this file. No other UI changes should be needed.
OPTIONAL_NUMERIC_KEYS = ["k_num_returns"]
OPTIONAL_SELECT_KEYS = ["k_subscription_status"]

WIDGET_DEFAULTS = {k: 0 for k in NUMERIC_KEYS + OPTIONAL_NUMERIC_KEYS}
WIDGET_DEFAULTS["k_satisfaction"] = 1
WIDGET_DEFAULTS.update({
    "k_complain": "No", "k_city_tier": 1, "k_login_device": "Phone",
    "k_payment_mode": "Debit Card", "k_gender": "Male",
    "k_order_cat": "Laptop & Accessory", "k_marital_status": "Single"
})

# All widget keys — used when "Clear Form" is pressed
ALL_WIDGET_KEYS = (NUMERIC_KEYS + OPTIONAL_NUMERIC_KEYS + ["k_satisfaction"]
                   + SELECT_KEYS + OPTIONAL_SELECT_KEYS)

# Initialize each widget's session state once, before the widgets are created
for k, default in WIDGET_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = default


def clear_form():
    for k in NUMERIC_KEYS + OPTIONAL_NUMERIC_KEYS:
        st.session_state[k] = 0
    st.session_state["k_satisfaction"] = 1
    for k in SELECT_KEYS + OPTIONAL_SELECT_KEYS:
        st.session_state[k] = None
    st.session_state.last_result = None


st.title("📊 E-Commerce Customer Churn Predictor")
st.caption("Deployed by Anshu Swarna")
st.write("Enter a customer's details below to predict their likelihood of churning.")

st.button("↺ Clear Form", on_click=clear_form)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1, key="k_tenure")
        warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", min_value=0, max_value=200, step=1,
                                             key="k_warehouse")
        delivery_time = st.number_input("Delivery Time Taken (days)", min_value=0, max_value=30, step=1,
                                         key="k_delivery_time")
        order_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=100, step=1,
                                      key="k_order_hike")
        hours_on_app = st.number_input("Hours Spent on App", min_value=0, max_value=10, step=1,
                                        key="k_hours_on_app")
        devices = st.number_input("Number of Devices Registered", min_value=0, max_value=10, step=1,
                                   key="k_devices")
        satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=10, key="k_satisfaction")
        addresses = st.number_input("Number of Addresses", min_value=0, max_value=20, step=1,
                                     key="k_addresses")

    with col2:
        complain = st.selectbox("Raised a Complaint Recently?", ["No", "Yes"],
                                 index=None, key="k_complain", placeholder="Select an option")
        coupon_used = st.number_input("Coupons Used (last month)", min_value=0, max_value=20, step=1,
                                       key="k_coupon_used")
        order_count = st.number_input("Order Count (last month)", min_value=0, max_value=20, step=1,
                                       key="k_order_count")
        days_since_order = st.number_input("Days Since Last Order", min_value=0, max_value=60, step=1,
                                            key="k_days_since_order")
        cashback = st.number_input("Average Cashback Amount (₹)", min_value=0, max_value=1000, step=10,
                                    key="k_cashback")
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=None, key="k_city_tier",
                                  placeholder="Select a tier")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        login_device = st.selectbox("Preferred Login Device", ["Phone", "Computer"],
                                     index=None, key="k_login_device", placeholder="Select a device")
        payment_mode = st.selectbox("Preferred Payment Mode",
                                     ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"],
                                     index=None, key="k_payment_mode", placeholder="Select a payment mode")
        gender = st.selectbox("Gender", ["Male", "Female"], index=None, key="k_gender",
                               placeholder="Select a gender")
    with col4:
        order_cat = st.selectbox("Preferred Order Category",
                                  ["Laptop & Accessory", "Mobiles", "Fashion", "Grocery", "Home Appliances","Others"],
                                  index=None, key="k_order_cat", placeholder="Select a category")
        marital_status = st.selectbox("Marital Status", ["Single", "Married"],
                                       index=None, key="k_marital_status", placeholder="Select a status")

    st.markdown("---")
    st.markdown("**Additional E-Commerce Details**")
    col5, col6 = st.columns(2)
    with col5:
        num_returns = st.number_input("Number of Returns / Refunds", min_value=0, max_value=50, step=1,
                                       key="k_num_returns")
    with col6:
        subscription_status = st.selectbox("Subscription Status", ["Active", "Inactive"],
                                            index=None, key="k_subscription_status", placeholder="Select a status")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    select_values = {
        "Complaint": complain, "City Tier": city_tier, "Preferred Login Device": login_device,
        "Preferred Payment Mode": payment_mode, "Gender": gender,
        "Preferred Order Category": order_cat, "Marital Status": marital_status
    }
    missing = [label for label, val in select_values.items() if val is None]

    if missing:
        st.warning(f"⚠️ Please fill in all fields before predicting. Missing: {', '.join(missing)}")
    else:
        customer = {
            'Tenure': float(tenure), 'WarehouseToHome': float(warehouse_to_home),
            'DeliveryTimeTaken': float(delivery_time),
            'OrderAmountHikeFromlastYear': float(order_hike), 'HourSpendOnApp': float(hours_on_app),
            'NumberOfDeviceRegistered': devices, 'SatisfactionScore': satisfaction,
            'NumberOfAddress': addresses, 'Complain': 1 if complain == "Yes" else 0,
            'CouponUsed': float(coupon_used), 'OrderCount': float(order_count),
            'DaySinceLastOrder': float(days_since_order), 'CashbackAmount': float(cashback),
            'CityTier': city_tier, 'PreferredLoginDevice': login_device, 'PreferredPaymentMode': payment_mode,
            'Gender': gender, 'PreferedOrderCat': order_cat, 'MaritalStatus': marital_status,
            'NumberOfReturns': num_returns,
            'SubscriptionStatus': subscription_status or "Not specified"
        }

        input_df = pd.DataFrame([customer])
        input_df[numerical_cols] = imputer.transform(input_df[numerical_cols])

        encoded = pd.DataFrame(encoder.transform(input_df[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))
        input_full = pd.concat([input_df.drop(columns=categorical_cols), encoded], axis=1)
        input_full[numerical_cols] = scaler.transform(input_full[numerical_cols])

        input_selected = input_full[selected_features]
        prediction = model.predict(input_selected)[0]
        proba_all = model.predict_proba(input_selected)[0]
        stay_probability = proba_all[0]
        churn_probability = proba_all[1]

        # ---- Log to session history ----
        st.session_state.history.insert(0, {
            "Tenure": tenure, "Complain": complain, "Satisfaction": satisfaction,
            "OrderCount": order_count, "Cashback": cashback, "DeliveryTime": delivery_time,
            "Prediction": "Churn" if prediction == 1 else "Stay",
            "Churn Prob": f"{churn_probability:.1%}"
        })
        st.session_state.history = st.session_state.history[:5]

        # ---- Persist this result so it survives the rerun triggered by the download button ----
        st.session_state.last_result = {
            "customer": customer,
            "prediction": int(prediction),
            "stay_probability": float(stay_probability),
            "churn_probability": float(churn_probability),
            "input_selected": input_selected,
        }


def build_pdf_report(customer, prediction, stay_probability, churn_probability, contribution=None):
    """Builds a one-page PDF churn prediction report and returns it as bytes."""
    pdf = FPDF()
    pdf.add_page()

    # ---- Header ----
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "E-Commerce Churn Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ---- Prediction result ----
    pdf.set_font("Helvetica", "B", 14)
    if prediction == 1:
        pdf.set_text_color(200, 30, 30)
        pdf.cell(0, 10, "Prediction: LIKELY TO CHURN", ln=True)
    else:
        pdf.set_text_color(30, 140, 30)
        pdf.cell(0, 10, "Prediction: LIKELY TO STAY", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Probability of Staying: {stay_probability:.1%}", ln=True)
    pdf.cell(0, 8, f"Probability of Churning: {churn_probability:.1%}", ln=True)
    pdf.ln(4)

    # ---- Customer details table ----
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Customer Details", ln=True)
    pdf.set_font("Helvetica", "", 10)

    label_w, value_w = 85, 95
    row_h = 7
    for key, value in customer.items():
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(label_w, row_h, str(key), border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(value_w, row_h, str(value), border=1, ln=True)

    # ---- Feature contribution ----
    if contribution is not None and len(contribution) > 0:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "What Drove This Prediction", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for feature, score in contribution.items():
            direction = "Toward Churn" if score > 0 else "Toward Stay"
            pdf.cell(0, 7, f"{feature}: {direction} (score {score:.3f})", ln=True)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 6, "Approximate importance-weighted contribution per feature (not full SHAP).")
        pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output(dest="S"))


# ---- Render the most recent prediction result (persists across reruns, e.g. after downloading) ----
if st.session_state.get("last_result"):
    r = st.session_state.last_result
    prediction = r["prediction"]
    stay_probability = r["stay_probability"]
    churn_probability = r["churn_probability"]
    input_selected = r["input_selected"]
    customer = r["customer"]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **CHURN**")
    else:
        st.success(f"✅ This customer is likely to **STAY**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probability of Staying", f"{stay_probability:.1%}")
    with col_b:
        st.metric("Probability of Churning", f"{churn_probability:.1%}")

    st.progress(min(float(churn_probability), 1.0))
    st.caption(f"The model is most confident this customer will "
               f"{'churn' if prediction == 1 else 'stay'}, "
               f"with a {max(stay_probability, churn_probability):.1%} confidence level.")

    # ---- Feature importance for this prediction (no matplotlib needed) ----
    contribution = None
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=selected_features)
        customer_values = input_selected.iloc[0]
        raw_contribution = (importances * customer_values).sort_values(key=abs, ascending=False)
        # Exclude gender-related features from the displayed drivers
        raw_contribution = raw_contribution[~raw_contribution.index.str.contains("Gender", case=False)]
        contribution = raw_contribution.head(8)

        st.markdown("#### What drove this prediction")
        for feature, score in contribution.items():
            direction = "🔴 Toward Churn" if score > 0 else "🟢 Toward Stay"
            bar_val = min(abs(float(score)), 1.0)
            st.markdown(f"**{feature}** — {direction}")
            st.progress(bar_val)
        st.caption("Approximate importance-weighted contribution per feature (not full SHAP).")

    # ---- Download this prediction as a PDF report ----
    pdf_bytes = build_pdf_report(customer, prediction, stay_probability, churn_probability, contribution)

    st.download_button(
        label="⬇️ Download Prediction as Report (PDF)",
        data=pdf_bytes,
        file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ---- Recent prediction history ----
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕘 Recent Predictions (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("📊 E-Commerce Churn Predictor | Deployed by Anshu Swarna")