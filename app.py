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
    This dataset contains over 1 million anonymized patient records related to COVID-19 cases. 
    It includes demographic data (age, sex) and a wide range of pre-existing comorbidities 
    such as diabetes, hypertension, and obesity. The data was provided by the instructor for MSIS 522.
    
    **The Prediction Task:**
    The target variable is **DEATH**, a binary indicator where 1 represents patient mortality 
    and 0 represents recovery.
    
    **The "So What":**
    This task is vital for resource allocation and triage. By predicting high-risk cases, 
    healthcare providers can prioritize ICU beds and critical care for those most likely 
    to face severe outcomes.
    """)

    st.subheader("Dataset Statistics & Features")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Basic Statistics:**")
        st.write("- Total Rows: 1,021,977")
        st.write("- Modeling Subset: 10,000 (Balanced)")
    with col_b:
        st.write("**Feature Details:**")
        st.write("- Total Features: 17")
        st.write("- Types: AGE (Numerical); 16 others (Categorical/Boolean)")

    st.subheader("Approach & Key Findings")
    st.write("""
    The original dataset was imbalanced (947,320 recoveries). To prevent bias, I performed 
    undersampling to create a balanced subset of 10,000 records. I tested five algorithms: 
    Logistic Regression, Decision Trees, Random Forests, LightGBM, and a Neural Network (MLP).
    
    **Key Finding:** The Neural Network performed best with an F1 score of 0.9068. While the 
    Neural Network showed strong predictive power, the tree-based ensemble models provided 
    a superior balance between accuracy and training efficiency.
    """)

# --- TAB 2: DESCRIPTIVE ANALYTICS ---
with tab2:
    st.header("Exploratory Data Analysis")

    st.subheader("Distribution of Patient Age")
    st.image("age_histogram.png") 
    st.write("""
    This histogram shows a wide distribution of ages, with a significant concentration of 
    patients in the 30–60 age range. The smooth line indicates that the data is relatively 
    continuous. Age is used as a primary quantitative predictor for risk stratification.
    """)

    st.subheader("Age Distribution by Mortality Outcome")
    st.image("age_boxplot.png")
    st.write("""
    The boxplot reveals an apparent difference; the median age for patients who died is 
    significantly higher than for those who survived. This means age is likely an important 
    feature, as older populations show a higher susceptibility to fatal outcomes.
    """)

    st.subheader("Impact of Pneumonia")
    st.image("pneumonia_comparison_bar_graph.png")
    st.write("""
    This visualization shows that patients with pre-existing pneumonia have a higher mortality 
    rate compared to those without. For clinical triage, this suggests that respiratory 
    health is a critical "red flag" that should trigger immediate medical intervention.
    """)

    st.subheader("Mortality Risk by Sex and Diabetes")
    st.image("mortality_risk_line_graph.png")
    st.write("""
    This point plot explores the interaction between biological sex and diabetes. It reveals 
    whether one gender with diabetes faces a higher risk than the other, providing a more 
    nuanced "risk combination" insight that a simple bar chart would miss.
    """)

    st.subheader("Feature Correlation Heatmap")
    st.image("patient_features_heatmap.png")
    st.write("""
    The heatmap reveals a strong positive correlation between HOSPITALIZED, PNEUMONIA, and DEATH. 
    These clinical indicators are the strongest predictors of mortality. Most simultaneous 
    presence of two diseases show low individual correlations, meaning they provide distinct, 
    non-redundant information to the model.
    """)

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.header("Model Evaluation & Comparison")

    st.subheader("Preprocessing & Documentation")
    st.write("""
    - **Feature Selection:** X defined as all patient features; y defined as the DEATH target.
    - **Train/Test Split:** 70/30 split (3,000 hold-out records) used to test generalization.
    - **Encoding:** Features were already binary; AGE (0-100) was maintained as-is. Tree-based models used are invariant to scaling.
    - **Missing Values:** Pre-cleaned; categorical unknowns were treated as distinct categories.
    """)

    st.subheader("Final Model Metrics")
    comparison_metrics = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "Neural Network"],
        "Accuracy": [0.8993, 0.8997, 0.8990, 0.8963, 0.8963],
        "F1 Score": [0.9039, 0.9059, 0.9053, 0.9024, 0.9008],
        "AUC-ROC": [0.9496, 0.9428, 0.9505, 0.9501, 0.9478]
    }
    st.dataframe(pd.DataFrame(comparison_metrics), use_container_width=True)

    st.divider()

    st.subheader("1. Model Comparison (F1 Score)")
    st.image("model_comparison_bar.png")
    st.write("The Logistic Regression model serves as our performance floor. With an F1 score of 0.9039, it provides a reliable starting point using linear relationships.")

    st.subheader("2. Decision Tree Analysis")
    st.image("best_decision_tree.png", caption="Logic Path")
    st.write("""
    HOSPITALIZED is the most significant initial splitter. Patients falling into the Right branch 
    (hospitalized) show a much higher immediate probability of death. For those hospitalized, 
    AGE is the next critical factor. For those not hospitalized, PNEUMONIA is the primary 
    secondary indicator.
    """)
    st.image("decision_tree_roc.png", caption="ROC Curve")
    st.write("Best Parameters: {'max_depth': 4, 'min_samples_split': 40}")

    st.subheader("3. Random Forest Analysis")
    st.image("random_forest_roc_curve.png")
    st.write("""
    The Random Forest achieved an AUC-ROC of 0.9505. By aggregating 200 trees, the model 
    reduces the variance found in the single Decision Tree, leading to more stable 
    and reliable predictions.
    """)

    st.subheader("4. LightGBM Analysis")
    st.image("lightgbm_roc_curve.png")
    st.write("""
    Best Parameters: {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 50}. 
    The high AUC-ROC score indicates the model is extremely effective at stratifying risk.
    """)

    st.subheader("5. Neural Network (MLP) Analysis")
    if os.path.exists("model_loss.png"):
        st.image("model_loss.png")
    if os.path.exists("model_accuracy.png"):
        st.image("model_accuracy.png")
    st.write("""
    The training history plots show a healthy convergence over 5 epochs. The lack of 
    a significant gap between training and validation metrics indicates the model is not overfitting.
    """)

    st.divider()
    st.subheader("Clinical Trade-offs")
    st.write("""
    A key trade-off exists between accuracy and interpretability. While the Neural Network 
    handles complex comorbidity interactions well, the Decision Tree offers transparent 
    "if-then" logic easier for professionals to audit.
    """)

# --- TAB 4: RISK PREDICTOR ---
with tab4:
    st.header("Explainability & Prediction")
    
    st.subheader("Global Explainability (SHAP)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if os.path.exists("shap_bar.png"): st.image("shap_bar.png")
    with col_s2:
        if os.path.exists("shap_beeswarm.png"): st.image("shap_beeswarm.png")
    
    st.write("""
    - **Strongest Impact:** HOSPITALIZED, AGE, and PNEUMONIA.
    - **Direction:** Hospitalization and increased Age strongly push predictions toward mortality.
    - **Clinical Utility:** Clinicians can use these to see that high risk is driven by 
      specific interactions, allowing for personalized care pathways.
    """)

    st.divider()
    st.subheader("Interactive Patient Risk Predictor")
    
    model_choice = st.selectbox("Select Model", ["LightGBM", "Random Forest", "Logistic Regression", "Neural Network", "Decision Tree"])

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
            prob = float(mlp_model.predict(user_input)[0])
            prediction = 1 if prob > 0.5 else 0
        else:
            selected = {"LightGBM": lgbm_model, "Random Forest": rf_model, "Logistic Regression": lr_model, "Decision Tree": dt_model}[model_choice]
            prediction = selected.predict(user_input)[0]
            prob = selected.predict_proba(user_input)[0][1]
        
        if prediction == 1: st.error(f"High Mortality Risk (Prob: {prob:.2%})")
        else: st.success(f"Recovery Likely (Prob: {prob:.2%})")

        if model_choice in ["LightGBM", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer({"LightGBM": lgbm_model, "Random Forest": rf_model, "Decision Tree": dt_model}[model_choice])
            user_shap_values = explainer(user_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(user_shap_values[0])
            st.pyplot(fig)
