import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
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

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.header("Project Overview")
    st.write("""
    This analysis is based on a comprehensive dataset of over 1 million anonymized patient records 
    related to COVID-19 cases. The data includes **17 diverse features** covering demographics and 
    a wide range of pre-existing comorbidities. 
    
    The primary prediction task is to determine **'DEATH'**—a binary outcome where 1 indicates 
    patient mortality and 0 represents recovery. This task is vital for resource allocation and triage.
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

    st.subheader("Approach & Key Findings")
    st.write("""
    Our modeling approach involved testing five distinct algorithms: Logistic Regression, 
    Decision Trees, Random Forests, LightGBM, and a Neural Network (MLP). Because the original 
    dataset was highly imbalanced toward survivors, we utilized undersampling to create a 
    balanced subset of 10,000 records. The Neural Network emerged as the top performer 
    with an F1 score of approximately 0.9068.
    """)

# --- TAB 2: DESCRIPTIVE ANALYTICS ---
with tab2:
    st.header("Visualizing the Data")
    st.write("Exploratory analysis reveals how demographics and comorbidities relate to patient outcomes.")

    st.subheader("Distribution of Patient Age")
    st.image("age_histogram.png") 
    st.write("This histogram shows a wide distribution of ages, with a significant concentration of patients in the 30–60 age range. Age is used as a primary quantitative predictor for risk stratification.")

    st.subheader("Age Distribution by Mortality Outcome")
    st.image("age_boxplot.png")
    st.write("The boxplot reveals that the median age for patients who died is significantly higher than for survivors. This confirms that older populations show a higher susceptibility to fatal outcomes.")

    st.subheader("Mortality Rate: Pneumonia vs. No Pneumonia")
    st.image("pneumonia_comparison_bar_graph.png")
    st.write("Patients with pre-existing pneumonia have a higher mortality rate compared to those without. For clinical triage, this suggests respiratory health is a critical 'red flag' for medical intervention.")

    st.subheader("Mortality Risk by Sex and Diabetes Status")
    st.image("mortality_risk_line_graph.png")
    st.write("This plot explores the interaction between biological sex and diabetes. It reveals whether specific gender and comorbidity combinations face higher risks, providing nuanced clinical insights.")

    st.subheader("Correlation Heatmap of Patient Features")
    st.image("patient_features_heatmap.png")
    st.write("The heatmap reveals a strong positive correlation between HOSPITALIZED, PNEUMONIA, and DEATH. These indicators are the strongest predictors of mortality within this dataset.")

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.header("Model Evaluation & Comparison")

    st.subheader("Final Test-Set Metrics")
    comparison_metrics = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "Neural Network"],
        "Accuracy": [0.8993, 0.8997, 0.8990, 0.8963, 0.9010],
        "F1 Score": [0.9039, 0.9059, 0.9053, 0.9024, 0.9084],
        "AUC-ROC": [0.9496, 0.9428, 0.9505, 0.9501, 0.9490]
    }
    st.dataframe(pd.DataFrame(comparison_metrics), use_container_width=True)

    st.divider()

    st.subheader("Visual Performance Analysis")

    # 1. Model Comparison Bar Chart
    st.image("model_comparison_bar.png", caption="F1 Score Comparison")
    st.write("The evaluation reveals that the Neural Network performed best, achieving the highest F1 score. While the Neural Network showed strong predictive power, the tree-based ensemble models provided a superior balance between accuracy and training efficiency.")

    # 2. Decision Tree Structure
    st.image("best_decision_tree.png", caption="Decision Tree Logic Path")
    st.write("The visualization shows that HOSPITALIZED is the most significant initial splitter. For those hospitalized, AGE becomes the next most critical factor, allowing clinicians to follow a clear 'if-then' logic.")

    # 3. Decision Tree ROC
    st.image("decision_tree_roc.png", caption="Decision Tree ROC Curve")
    st.write("The Decision Tree ROC curve shows robust discriminatory power. While slightly lower than ensemble methods, it offers transparency that is easier for clinical professionals to trust and audit.")

    # 4. Random Forest ROC
    st.image("random_forest_roc_curve.png", caption="Random Forest ROC Curve")
    st.write("The Random Forest model achieved an AUC-ROC of 0.9505. By aggregating 200 trees, the model reduces variance found in single trees, leading to more stable and reliable predictions.")

    # 5. LightGBM ROC
    st.image("lightgbm_roc_curve.png", caption="LightGBM ROC Curve")
    st.write("The high AUC-ROC score for LightGBM indicates the model is extremely effective at stratifying patient risk. It performs exceptionally well on tabular data with complex feature interactions.")

    # 6. Neural Network Training
    if os.path.exists("model_loss.png"):
        st.image("model_loss.png", caption="Model Loss History")
    if os.path.exists("model_accuracy.png"):
        st.image("model_accuracy.png", caption="Model Accuracy History")
    st.write("The training history plots show a healthy convergence, with both training and validation loss decreasing steadily. The lack of a significant gap indicates the model is not overfitting.")

    st.divider()
    st.subheader("Optimized Hyperparameters")
    st.markdown("""
    * **Decision Tree**: `max_depth: 4`, `min_samples_split: 40`
    * **Random Forest**: `max_depth: 8`, `n_estimators: 200`
    * **LightGBM**: `learning_rate: 0.05`, `max_depth: 4`, `n_estimators: 50`
    """)

# --- TAB 4: RISK PREDICTOR ---
with tab4:
    st.header("Interactive Risk Predictor")
    
    st.subheader("Global Model Explainability")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.image("shap_bar.png", caption="Global Feature Importance")
    with col_s2:
        st.image("shap_beeswarm.png", caption="SHAP Beeswarm Plot")

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
        sex_input = st.selectbox("Sex (0=Male, 1=Female)", [0, 1])

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
            selected_model = {"LightGBM": lgbm_model, "Random Forest": rf_model, 
                              "Logistic Regression": lr_model, "Decision Tree": dt_model}[model_choice]
            prediction = selected_model.predict(user_input)[0]
            prob = selected_model.predict_proba(user_input)[0][1]
        
        if prediction == 1:
            st.error(f"High Mortality Risk (Probability: {prob:.2%})")
        else:
            st.success(f"Recovery Likely (Mortality Probability: {prob:.2%})")

        st.subheader(f"Why did the {model_choice} model make this prediction?")
        if model_choice in ["LightGBM", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(selected_model)
            user_shap_values = explainer(user_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(user_shap_values[0])
            st.pyplot(fig)
        else:
            st.info("Waterfall plots are best visualized for the tree-based models in this application.")
