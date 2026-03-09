import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tensorflow.keras.models import load_model

# --- LOAD SAVED MODELS ---
@st.cache_resource
def load_all_models():
    lgbm = joblib.load('lightgbm_model.pkl')
    rf = joblib.load('random_forest_model.pkl')
    lr = joblib.load('logistic_regression_model.pkl')
    dt = joblib.load('decision_tree_model.pkl')
    mlp = load_model('mlp_model.keras')
    return lgbm, rf, lr, dt, mlp

lgbm_model, rf_model, lr_model, dt_model, mlp_model = load_all_models()

st.title("COVID-19 Patient Mortality Risk Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", "Descriptive Analytics", 
    "Model Performance", "Risk Predictor"
])

# TAB 1: EXECUTIVE SUMMARY
with tab1:
    st.header("Project Overview")
    
    st.write("""
    This analysis is based on a comprehensive dataset of over 1 million anonymized patient records 
    related to COVID-19 cases. 
    The data includes **17 diverse features** covering demographics and a wide range of 
    pre-existing comorbidities. 
    
    The primary prediction task is to determine **'DEATH'**—a binary outcome where 1 indicates 
    patient mortality and 0 represents recovery.
    """)

    st.subheader("Dataset Statistics & Features")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Data Volume:**")
        st.write("- Total Rows: 1,021,977")
        st.write("- Modeling Subset: 10,000 (Balanced)")
    with col_b:
        st.write("**Feature Types:**")
        st.write("- **Numerical:** AGE")
        st.write("- **Categorical/Boolean:** 16 features (e.g., SEX, PNEUMONIA, DIABETES, HOSPITALIZED)")

    st.subheader("The 'So What'")
    st.write("""
    In a high-pressure healthcare environment, the ability to accurately stratify patient risk 
    is a matter of life and death. By leveraging predictive analytics, healthcare systems can 
    optimize resource allocation and triage efforts. These insights allow providers 
    to prioritize intensive care unit (ICU) beds and critical interventions for patients 
    statistically most likely to face severe outcomes, ultimately improving clinical 
    efficiency and patient survival rates.
    """)

    st.subheader("Approach & Key Findings")
    st.write("""
    Our modeling approach involved testing five distinct algorithms: Logistic Regression, 
    Decision Trees, Random Forests, LightGBM, and a Neural Network (MLP). 
    Because the original dataset was highly imbalanced toward survivors (over 947,000 recoveries), 
    we utilized undersampling to create a balanced subset of 10,000 records for robust training. 
    The Neural Network emerged as the top performer with an F1 score of 0.9084.
    """)

# TAB 2: DESCRIPTIVE ANALYTICS
with tab2:
    st.header("Visualizing the Data")
    st.write("The original dataset was highly imbalanced. To ensure the model learns the characteristics of both outcomes equally, I performed undersampling to create a balanced subset of 10,000 patient records.")

    st.subheader("Distribution of Patient Age")
    st.image("age_histogram.png") 
    st.write("This histogram shows a wide distribution of ages, with a significant concentration of patients in the 30–60 age range. Age is used as a primary quantitative predictor for risk stratification.")

    st.subheader("Age Distribution by Mortality Outcome")
    st.image("age_boxplot.png")
    st.write("The boxplot reveals that the median age for patients who died is significantly higher than for those who survived. This means age is likely an important feature for susceptibility.")

    st.subheader("Mortality Rate: Pneumonia vs. No Pneumonia")
    st.image("pneumonia_comparison_bar_graph.png")
    st.write("This visualization shows that patients with pre-existing pneumonia have a higher mortality rate compared to those without. This suggests respiratory health is a critical 'red flag'.")

    st.subheader("Mortality Risk by Sex and Diabetes")
    st.image("mortality_risk_line_graph.png")
    st.write("This point plot explores the interaction between biological sex and diabetes, providing a more nuanced 'risk combination' insight than a simple bar chart.")

    st.subheader("Correlation Heatmap of Patient Features")
    st.image("patient_features_heatmap.png")
    st.write("The heatmap reveals a strong positive correlation between HOSPITALIZED, PNEUMONIA, and DEATH. These clinical indicators are the strongest predictors of mortality.")

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.header("Model Evaluation & Comparison")

    st.subheader("Final Test-Set Metrics")
    st.write("""
    All models were evaluated on a held-out test set of 3,000 patients. The Neural Network achieved the highest F1 score, 
    making it the most balanced model for this specific healthcare task.
    """)
    
    comparison_metrics = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "Neural Network"],
        "Accuracy": [0.8993, 0.8997, 0.8990, 0.8963, 0.9010],
        "Precision": [0.8847, 0.8718, 0.8712, 0.8720, 0.8634],
        "Recall": [0.9239, 0.9427, 0.9421, 0.9349, 0.9584],
        "F1 Score": [0.9039, 0.9059, 0.9053, 0.9024, 0.9084],
        "AUC-ROC": [0.9496, 0.9428, 0.9505, 0.9501, 0.9490]
    }
    st.dataframe(pd.DataFrame(comparison_metrics))

    st.subheader("Visual Performance Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.image("model_comparison_bar.png", caption="F1 Score Comparison")
        st.image("random_forest_roc.png", caption="Random Forest ROC Curve")

    with col2:
        st.image("nn_training_history.png", caption="Neural Network Loss & Accuracy")
        st.image("lightgbm_roc.png", caption="LightGBM ROC Curve")

    st.subheader("Optimized Hyperparameters")
    st.markdown("""
    * **Decision Tree**: `max_depth: 4`, `min_samples_split: 40`
    * **Random Forest**: `max_depth: 8`, `n_estimators: 200`
    * **LightGBM**: `learning_rate: 0.05`, `max_depth: 4`, `n_estimators: 50`
    """)

    st.subheader("Interpretability vs. Accuracy")
    st.write("""
    A key trade-off observed is between accuracy and interpretability. While ensemble methods and the 
    Neural Network are more accurate, the Decision Tree offers a transparent 'if-then' logic.
    """)

# TAB 4: RISK PREDICTOR
with tab4:
    st.header("Interactive Risk Predictor")
    
    st.subheader("Global Model Explainability")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.image("shap_bar.png", caption="Feature Importance (Mean Absolute SHAP)")
    with col_s2:
        st.image("shap_beeswarm.png", caption="Beeswarm Plot (Feature Impact & Direction)")

    st.divider()

    st.subheader("Predict Patient Risk")
    
    model_choice = st.selectbox("Select Model for Prediction", 
                                ["LightGBM", "Random Forest", "Logistic Regression", "Neural Network", "Decision Tree"])

    c1, c2, c3 = st.columns(3)
    with c1:
        age_input = st.slider("Age", 0, 100, 50)
        hosp_input = st.selectbox("Hospitalized", [0, 1])
    with c2:
        pneu_input = st.selectbox("Pneumonia", [0, 1])
        covid_input = st.selectbox("COVID Positive", [0, 1])
    with c3:
        diab_input = st.selectbox("Diabetes", [0, 1])
        sex_input = st.selectbox("Sex (0=M, 1=F)", [0, 1])

    user_input = pd.DataFrame([[sex_input, hosp_input, pneu_input, age_input, 0, diab_input, 0, 0, 0, 0, 0, 0, 0, 0, 0, covid_input]], 
                               columns=['SEX', 'HOSPITALIZED', 'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 
                                        'IMMUNOSUPPRESSION', 'HYPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                                        'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'COVID_POSITIVE'])

    if st.button("Run Prediction"):
        if model_choice == "Neural Network":
            selected_model = mlp_model
            prob = float(selected_model.predict(user_input)[0])
            prediction = 1 if prob > 0.5 else 0
        else:
            if model_choice == "LightGBM":
                selected_model = lgbm_model
            elif model_choice == "Random Forest":
                selected_model = rf_model
            elif model_choice == "Logistic Regression":
                selected_model = lr_model
            else:
                selected_model = dt_model
            
            prediction = selected_model.predict(user_input)[0]
            prob = selected_model.predict_proba(user_input)[0][1]
        
        if prediction == 1:
            st.error(f"Predicted Outcome: High Mortality Risk (Probability: {prob:.2%})")
        else:
            st.success(f"Predicted Outcome: Recovery (Probability of Mortality: {prob:.2%})")

        st.subheader(f"Why did the {model_choice} model make this prediction?")
        
        if model_choice in ["LightGBM", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(selected_model)
            user_shap_values = explainer(user_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(user_shap_values[0])
            st.pyplot(fig)
        else:
            st.info("SHAP Waterfall plots are best visualized for the tree-based models in this app.")