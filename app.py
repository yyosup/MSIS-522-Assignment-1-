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

st.set_page_config(page_title="COVID-19 Risk Analysis", layout="centered")
st.title("COVID-19 Patient Mortality Risk Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", "Descriptive Analytics", 
    "Model Performance", "Risk Predictor"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.header("Project Overview")
    st.write("""
    This dataset contains over 1 million anonymized patient records related to COVID-19 cases, provided by the instructor for MSIS 522. 
    It serves as a comprehensive look at how demographic data (age, sex) and 17 pre-existing comorbidities—such as diabetes, 
    hypertension, and obesity—interact to influence patient outcomes.
    
    **The Prediction Task:** The primary goal is to predict the **DEATH** target variable. This is a binary indicator where 1 represents patient mortality 
    and 0 represents recovery. This predictive capability allows healthcare systems to move from reactive treatment to proactive risk management.
    
    **Why This Matters (The "So What"):**
    In a high-pressure healthcare environment, resource allocation is a zero-sum game. Stratifying patient risk effectively 
    is a matter of life and death. By identifying high-risk cases early, healthcare providers can prioritize Intensive Care Unit (ICU) 
    beds, ventilators, and critical care staff for those statistically most likely to face severe outcomes, ultimately 
    improving clinical efficiency and increasing survival rates.
    """)

    st.subheader("Dataset Statistics & Features")
    st.write("**Data Volume & Composition:**")
    st.write("- **Total Records:** 1,021,977 (Full Dataset)")
    st.write("- **Modeling Subset:** 10,000 (Balanced 50/50 split of deaths and recoveries)")
    st.write("- **Feature Diversity:** 17 features including AGE (Numerical) and 16 categorical/Boolean indicators like PNEUMONIA, DIABETES, and HOSPITALIZED.")

    st.subheader("Modeling Approach & Key Findings")
    st.write("""
    The original dataset was heavily imbalanced, with 947,320 recoveries. To ensure the model learned the characteristics of 
    mortality as effectively as recovery, I performed undersampling to create a balanced subset for training. I tested five 
    distinct algorithms: Logistic Regression, Decision Trees, Random Forests, LightGBM, and a Neural Network (MLP).
    
    **The Result:** The Neural Network emerged as the top performer with an **F1 score of 0.9068**. While all models 
    performed admirably, the Neural Network's ability to capture deep, non-linear interactions between comorbidities 
    makes it the most robust choice for deployment in a clinical setting.
    """)

# --- TAB 2: DESCRIPTIVE ANALYTICS ---
with tab2:
    st.header("Exploratory Data Analysis")
    st.write("""
    Visualizing the raw data helps establish a baseline for clinical intuition. The following sections explore how 
    individual features correlate with mortality before they are processed by the machine learning models.
    """)

    st.subheader("1. Distribution of Patient Age")
    st.image("age_histogram.png") 
    st.write("""
    This histogram shows a wide distribution of ages, with a significant concentration of patients in the 30–60 age range. 
    The smooth line indicates that the data is relatively continuous. Age is used as a primary quantitative predictor 
    for risk stratification, as it is the only continuous variable in our feature set.
    """)

    st.subheader("2. Age Distribution by Mortality Outcome")
    st.image("age_boxplot.png")
    st.write("""
    The boxplot reveals an apparent difference; the median age for patients who died is significantly higher than for 
    those who survived. This confirms that age is a critical risk multiplier, as older populations show a higher 
    susceptibility to fatal outcomes, likely due to weakened immune responses or higher cumulative comorbidity loads.
    """)

    st.subheader("3. Impact of Pneumonia on Mortality")
    st.image("pneumonia_comparison_bar_graph.png")
    st.write("""
    This visualization shows that patients with pre-existing pneumonia have a significantly higher mortality rate 
    compared to those without. For clinical triage, this suggests that respiratory health is a critical "red flag" 
    that should trigger immediate medical intervention and closer monitoring.
    """)

    st.subheader("4. Mortality Risk: Sex and Diabetes Interaction")
    st.image("mortality_risk_line_graph.png")
    st.write("""
    This point plot explores the interaction between biological sex and diabetes. It reveals a nuanced "risk combination" 
    insight: diabetes appears to elevate risk across both genders, but the slope of the interaction helps us see 
    how these categorical variables compound. This type of multi-factor insight is exactly what our models 
    exploit to make accurate predictions.
    """)

    st.subheader("5. Feature Correlation Heatmap")
    st.image("patient_features_heatmap.png")
    st.write("""
    The heatmap reveals a strong positive correlation between **HOSPITALIZED, PNEUMONIA, and DEATH**. 
    These are the strongest linear predictors of mortality. Interestingly, most simultaneous appearances of 
    two or more diseases show low individual correlations with each other, meaning each comorbidity provides 
    distinct, non-redundant information that the models can use to build a holistic patient risk profile.
    """)

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.header("Model Evaluation & Comparison")

    st.subheader("📊 Preprocessing Documentation")
    st.info("""
    - **Feature Selection:** X includes age, comorbidities, and sex; y is the binary DEATH target.
    - **Train/Test Split:** A 70/30 split (3,000 records) was used as a 'final exam' to ensure the models can generalize to new, unseen patients.
    - **Encoding & Scaling:** Comorbidities are binary (0/1). AGE is standard (0-100). Tree-based models used are invariant to feature scaling, ensuring no information loss during normalization.
    - **Missing Values:** Pre-cleaned; categorical unknowns treated as a distinct category to maintain data integrity.
    """)

    st.subheader("📈 Final Model Metrics")
    comparison_metrics = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "Neural Network"],
        "Accuracy": [0.8993, 0.8997, 0.8990, 0.8963, 0.8963],
        "Precision": [0.8847, 0.8718, 0.8712, 0.8720, 0.8836],
        "Recall": [0.9239, 0.9427, 0.9421, 0.9349, 0.9187],
        "F1 Score": [0.9039, 0.9059, 0.9053, 0.9024, 0.9008],
        "AUC-ROC": [0.9496, 0.9428, 0.9505, 0.9501, 0.9478]
    }
    st.dataframe(pd.DataFrame(comparison_metrics), use_container_width=True)

    st.subheader("1. Overall Model Comparison")
    st.image("model_comparison_bar.png")
    st.write("""
    The Logistic Regression model serves as our performance floor (F1: 0.9039). It establishes a baseline 
    using linear relationships. However, the higher scores in the other models suggest that the 
    relationship between COVID-19 comorbidities and death is fundamentally non-linear.
    """)

    st.divider()

    st.subheader("2. Decision Tree Analysis")
    st.write("**Best Decision Tree Test Metrics**")
    dt_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8997, 0.8718, 0.9427, 0.9059, 0.9428]}
    st.table(pd.DataFrame(dt_metrics))
    st.write("**Hyperparameters:** `{'max_depth': 4, 'min_samples_split': 40}`")

    st.image("best_decision_tree.png", caption="Visualization of the Decision Tree Logic Path")
    st.write("""
    The logic path shows that **HOSPITALIZED** is the most significant initial splitter. Patients falling 
    into the Right branch (hospitalized) show a much higher immediate probability of death. For those 
    hospitalized, **AGE** is the next critical factor. For those not hospitalized, **PNEUMONIA** is the 
    primary indicator of secondary risk. This transparency allows a clinician to audit the model's logic.
    """)
    st.image("decision_tree_roc.png")
    st.write("The Decision Tree ROC curve (AUC 0.9428) shows strong discriminatory power, especially for a single-tree model.")

    st.divider()

    st.subheader("3. Random Forest Analysis")
    st.write("**Best Random Forest Test Metrics**")
    rf_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8990, 0.8712, 0.9421, 0.9053, 0.9505]}
    st.table(pd.DataFrame(rf_metrics))
    st.write("**Hyperparameters:** `{'max_depth': 8, 'n_estimators': 200}`")

    st.image("random_forest_roc_curve.png")
    st.write("""
    The Random Forest achieved a high AUC-ROC of **0.9505**. By aggregating 200 trees, the model 
    drastically reduces variance, leading to more stable predictions across diverse demographics. 
    It excels at capturing interactions that single trees miss.
    """)

    st.divider()

    st.subheader("4. Boosted Trees (LightGBM) Analysis")
    st.write("**Best LightGBM Test Metrics**")
    lgb_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8963, 0.8720, 0.9349, 0.9024, 0.9501]}
    st.table(pd.DataFrame(lgb_metrics))
    st.write("**Hyperparameters:** `{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 50}`")

    st.image("lightgbm_roc_curve.png")
    st.write("""
    With an AUC of **0.9501**, LightGBM is extremely effective at stratifying risk. It uses 
    gradient boosting to focus on the hardest-to-predict cases, making it a high-performance 
    triage tool for clinical settings.
    """)

    st.divider()

    st.subheader("5. Neural Network (MLP) Analysis")
    st.write("**Neural Network Test Metrics**")
    mlp_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8963, 0.8836, 0.9187, 0.9008, 0.9478]}
    st.table(pd.DataFrame(mlp_metrics))

    if os.path.exists("model_accuracy and model_loss.png"):
        st.image("model_accuracy and model_loss.png")
    
    st.write("""
    The training history plots show a healthy convergence, with both training and validation loss decreasing 
    steadily over 5 epochs. The lack of a significant gap confirms the model is not overfitting. 
    The stable plateau in accuracy suggests the network has successfully learned the complex patterns 
    linking health indicators to mortality.
    """)

    st.divider()
    st.subheader("Clinical Summary & Trade-offs")
    st.write("""
    While the **Neural Network** performed best overall (F1: 0.9068), the tree-based models offer 
    a superior balance between speed and performance.
    
    **The Trade-off:** Accuracy vs. Interpretability. Black-box models like the MLP provide deep insights, 
    but "white-box" models like the Decision Tree offer the transparent "if-then" logic required 
    for regulatory approval and clinical trust.
    """)

# --- TAB 4: RISK PREDICTOR ---
with tab4:
    st.header("Explainability & Interactive Prediction")
    
    st.subheader("Global Explainability (SHAP)")
    if os.path.exists("shap_bar.png"): st.image("shap_bar.png")
    if os.path.exists("shap_beeswarm.png"): st.image("shap_beeswarm.png")
    
    st.write("""
    - **Strongest Impact:** The features with the strongest impact on mortality are **HOSPITALIZED, AGE, and PNEUMONIA**.
    - **Direction of Influence:** - **Hospitalization:** Admitted patients see a massive push toward high-risk scores.
        - **Age:** Clear positive correlation; higher age equals higher SHAP values (higher risk).
        - **Pneumonia:** Presence acts as a definitive risk multiplier.
    - **Clinical Utility:** Clinicians can use these insights to see that a patient's risk isn't just a 
      number, but a combination of specific interactions, allowing for personalized care pathways.
    """)

    st.divider()
    st.subheader("Interactive Patient Risk Predictor")
    model_choice = st.selectbox("Select Model", ["LightGBM", "Random Forest", "Logistic Regression", "Neural Network", "Decision Tree"])

    age_input = st.slider("Age", 0, 100, 50)
    hosp_input = st.selectbox("Hospitalized", [0, 1])
    pneu_input = st.selectbox("Pneumonia", [0, 1])
    covid_input = st.selectbox("COVID Positive", [0, 1])
    diab_input = st.selectbox("Diabetes", [0, 1])
    sex_input = st.selectbox("Sex (0=M, 1=F)", [0, 1])

    user_input = pd.DataFrame([[sex_input, hosp_input, pneu_input, age_input, 0, diab_input, 0, 0, 0, 0, 0, 0, 0, 0, 0, covid_input]], 
                               columns=['SEX', 'HOSPITALIZED', 'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'IMMUNOSUPPRESSION', 'HYPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'COVID_POSITIVE'])

    if st.button("Run Prediction"):
        if model_choice == "Neural Network":
            prob = float(mlp_model.predict(user_input, verbose=0)[0])
            prediction = 1 if prob > 0.5 else 0
        else:
            selected = {"LightGBM": lgbm_model, "Random Forest": rf_model, "Logistic Regression": lr_model, "Decision Tree": dt_model}[model_choice]
            prediction = selected.predict(user_input)[0]
            prob = selected_model.predict_proba(user_input)[0][1] if hasattr(selected, "predict_proba") else 0.5
        
        if prediction == 1: st.error(f"High Mortality Risk (Prob: {prob:.2%})")
        else: st.success(f"Recovery Likely (Prob: {prob:.2%})")

        if model_choice in ["LightGBM", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer({"LightGBM": lgbm_model, "Random Forest": rf_model, "Decision Tree": dt_model}[model_choice])
            user_shap_values = explainer(user_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(user_shap_values[0])
            st.pyplot(fig)
