import streamlit as st
import numpy as np
import pandas as pd
import urllib.request
import joblib

# -----------------------------
# Load model + preprocessing
# -----------------------------

def load_model(url, filename):
    urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

final_svm = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_2/main/svm_model.pkl",
    "svm_model.pkl"
)
preprocessor_all = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_2/main/m2_preprocessor.pkl",
    "m2_preprocessor.pkl"
)
le = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_2/main/m2_label_encoder_kind_group.pkl",
    "m2_label_encoder_kind_group.pkl"
)


# -----------------------------
# Mappings
# -----------------------------
age_mapping = {
    'Under 16': 1,
    '16-19': 2,
    '20-24': 3,
    '25-34': 4,
    '35-44': 5,
    '45-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65+': 9,
    'Unknown': 0
}

X_train_columns = ['age_band', 'main_activity_Construction of buildings',
       'main_activity_Specialised activities', 'riskcat_Assault',
       'riskcat_Burns from hot substances/surfaces',
       'riskcat_Chemical harm, irritant or corrosive',
       'riskcat_Confined Spaces', 'riskcat_Crushed by excavation',
       'riskcat_Electric Shock', 'riskcat_Electric shock',
       'riskcat_Fall from ladder', 'riskcat_Fall from open edge',
       'riskcat_Fall from scaffold', 'riskcat_Fall through fragile material',
       'riskcat_Fire/explosion', 'riskcat_MEWP operations',
       'riskcat_Machinery guarding',
       'riskcat_Materials Handling including Manual handling',
       'riskcat_Mechanical Lifting Operations',
       'riskcat_Mechanical lifting operations', 'riskcat_Other',
       'riskcat_Other - episode of illness at work',
       'riskcat_Other - infection acquired at work',
       'riskcat_Other - injury whilst driving plant',
       'riskcat_Other - road traffic accident', 'riskcat_Other - rope access',
       'riskcat_Overturning plant or moving machinery',
       'riskcat_Public protection', 'riskcat_Slip or trip on same level',
       'riskcat_Struck by falling object', 'riskcat_Struck by flying object',
       'riskcat_Struck by moving vehicle', 'riskcat_Unintended collapse',
       'riskcat_Using hand/power tools', 'Region_Eastern', 'Region_London',
       'Region_North East', 'Region_North West', 'Region_Scotland',
       'Region_South East', 'Region_South West', 'Region_Wales',
       'Region_West Midlands', 'Region_Yorkshire and the Humber', 'month_sin',
       'month_cos']


# -----------------------------
# UI
# -----------------------------
st.title("🚨 Accident Mechanism Prediction")

main_activity = st.selectbox("Main Activity", [
    'Construction of buildings', 'Civil engineering', 'Specialised activities'
])

riskcat = st.selectbox("Risk Category", ['Slip or trip on same level',
       'Materials Handling including Manual handling',
       'Fall from scaffold', 'Struck by flying object',
       'Fall from ladder', 'Using hand/power tools',
       'Fall from open edge', 'Chemical harm, irritant or corrosive',
       'Struck by falling object', 'MEWP operations',
       'Mechanical Lifting Operations', 'Public protection',
       'Struck by moving vehicle', 'Machinery guarding', 'Electric shock',
       'Overturning plant or moving machinery',
       'Fall through fragile material', 'Fire/explosion',
       'Burns from hot substances/surfaces', 'Assault',
       'Unintended collapse', 'Other - road traffic accident', 'Other',
       'Confined Spaces', 'Other - infection acquired at work',
       'Other - rope access',
       'Other - injury whilst driving plant',
       'Other - episode of illness at work',
       'Mechanical lifting operations', 'Crushed by excavation'])

region = st.selectbox("Region", ['London', 'South East', 'Scotland', 'North West', 'South West', 'East Midlands', 'Eastern', 'North East', 'Wales', 'West Midlands', 'Yorkshire and the Humber']
)

age_band = st.selectbox("Age Band", ['Under 16', '16-19', '20-24', '25-34', '35-44', '45-54', '55-59', '60-64', '65+', 'Unknown'])
month = st.selectbox("Month", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])


# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    ## Step 1: build input DataFrame (RAW format)
    input_data = {
        "main_activity": main_activity,
        "riskcat": riskcat,
        "Region": region,
        "age_band": age_band,
        "month": month
    }


    feature_cols = X_train_columns  # must be saved before training

    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0  # initialize all zeros


    ## Mapping
    row['age_band'] = age_mapping[input_data['age_band']]

    month_val = int(input_data["month"])

    row['month_sin'] = np.sin(2 * np.pi * month_val / 12)
    row['month_cos'] = np.cos(2 * np.pi * month_val / 12)


    # helper function
    def set_feature(prefix, value):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row[col] = 1

    set_feature("main_activity", input_data["main_activity"])
    set_feature("riskcat", input_data["riskcat"])
    set_feature("Region", input_data["Region"])


    # Step 2: preprocess
    X_input = preprocessor_all.transform(row)

    # Step 3: prediction
    prediction = final_svm.predict(X_input)
    proba = max(final_svm.predict_proba(X_input)[0])

    # Step 4: decode label
    pred_label = le.inverse_transform([prediction[0]])
    confidence = proba * 100

    # -----------------------------
    # Output
    # -----------------------------
    st.success(f"Prediction: {pred_label[0]}")
    st.info(f"Confidence: {confidence:.2f}%")
