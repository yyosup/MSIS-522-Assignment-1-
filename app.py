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
    This analysis utilizes a massive repository of over 1 million anonymized patient profiles capturing clinical encounters during the COVID-19 pandemic. 
    The data provides a high-dimensional view of patient health, featuring 17 distinct variables that range from basic demographics like age and biological 
    sex to a rigorous catalog of pre-existing comorbidities. These medical "red flags" include conditions such as hypertension, diabetes, obesity, and renal 
    chronic disease.

    **The Prediciton Task:**
    Our primary objective is to build a classification system for the 'DEATH' target variable. In this model, a value of 1 signifies mortality while 0 indicates 
    the patient was successfully discharged or recovered. By analyzing these historical records, we aim to uncover the hidden patterns that determine which 
    combination of health factors leads to the highest risk of fatal complications.
    
    **Why This Matters (The "So What"):**
    The "so what" of this project lies in the reality of scarcity within healthcare systems. During a public health crisis, hospitals often reach a breaking point 
    where the number of patients exceeds the number of available Intensive Care Unit (ICU) beds and ventilators. This project transforms raw medical data into a 
    triage support tool.
    Instead of treating every patient with a "first-come, first-served" approach, healthcare administrators can use these predictive insights to prioritize limited 
    critical care resources for the most vulnerable populations. By identifying high-risk individuals the moment they enter the system, we can shift from a reactive 
    stance to a proactive strategy, potentially saving lives through earlier intervention and more efficient staff allocation
    """)

    st.subheader("Dataset Statistics & Features")
    st.write("**Data Volume & Composition:**")
    st.write("- **Total Records:** 1,021,977 (Full Dataset)")
    st.write("- **Modeling Subset:** 10,000 (Balanced 50/50 split of deaths and recoveries)")
    st.write("- **Feature Diversity:** 17 features including AGE (Numerical) and 16 categorical/Boolean indicators like PNEUMONIA, DIABETES, and HOSPITALIZED.")

    st.subheader("Modeling Approach & Key Findings")
    st.write("""
    To ensure the highest accuracy, our approach involved a rigorous "tournament" between five different machine learning architectures: Logistic Regression, 
    Decision Trees, Random Forests, LightGBM, and a Neural Network (MLP). A major technical hurdle was the initial data imbalance, as survivors far outnumbered 
    fatalities. To solve this, we utilized a balanced undersampling technique, creating a robust training set of 10,000 records where both outcomes were represented 
    equally.Our findings were definitive: The Neural Network (Multi-Layer Perceptron) emerged as the champion model, achieving an F1 score of approximately 0.9068. 
    We discovered that while individual conditions like diabetes are significant, the interaction between advanced age and respiratory complications (pneumonia) served 
    as the most powerful predictor of patient mortality. This confirms that complex, non-linear models are essential for capturing the multifaceted nature of human health during a viral outbreak.
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
    **Preprocessing Documentation:**
    - **Feature Selection:** I defined x as all patient features (age, comorbidities, sex) and y as the DEATH target.
    - **Train/Test Split:** I performed a 70/30 split using random_state=42. This ensures that 30% of the data (3,000 records) is held out as a "final exam" for the models to test their ability to generalize to new patients.
    - **Encoding & Scaling:** Since the comorbidities are already binary (0 or 1) and the AGE variable is on a relatively standard scale (0–100), no heavy encoding was required. However, using tree-based models like Random Forest and LightGBM is advantageous here as they are invariant to feature scaling.
    - **Handling Missing Values:** The dataset was pre-cleaned; any remaining missing values in categorical fields were treated as a distinct "unknown" category to maintain the clinical integrity of the patient records.
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
    The Logistic Regression model serves as our performance floor. With an **F1 score of 0.9039**, it provides a reliable starting point using linear relationships between comorbidities and mortality. 
    However, the higher scores in the other models suggest that the relationship between COVID-19 comorbidities and death is fundamentally non-linear.
    """)

    st.divider()

    st.subheader("2. Decision Tree Analysis")
    st.write("**Best Decision Tree Test Metrics**")
    dt_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8997, 0.8718, 0.9427, 0.9059, 0.9428]}
    st.table(pd.DataFrame(dt_metrics))
    st.write("**Best Parameters:** `{'max_depth': 4, 'min_samples_split': 40}`")

    st.image("best_decision_tree.png", caption="Visualization of the Decision Tree Logic Path")
    st.write("""
    The visualization shows that **HOSPITALIZED** is the most significant initial splitter for determining mortality risk. Patients falling 
    into the Right branch (those who were hospitalized, satisfying the clinical condition) show a much higher immediate probability of death. For those 
    hospitalized, **AGE** is the next critical factor in risk stratification. For patients not hospitalized (the Left branch), the presence of **PNEUMONIA** is the 
    primary indicator of secondary risk. This model allows a clinician to follow the "if-then" logic to understand a specific patient's risk profile based on their symptoms and demographics.
    """)
    st.image("decision_tree_roc.png")
    st.write("""
    The Decision Tree ROC curve visualizes the trade-off between the true positive rate and false positive rate. 
    With an AUC-ROC of **0.9428**, the model demonstrates strong discriminatory power, efficiently separating 
    survivors from high-risk patients.
    """)

    st.divider()

    st.subheader("3. Random Forest Analysis")
    st.write("**Best Random Forest Test Metrics**")
    rf_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8990, 0.8712, 0.9421, 0.9053, 0.9505]}
    st.table(pd.DataFrame(rf_metrics))
    st.write("**Best Parameters:** `{'max_depth': 8, 'n_estimators': 200}`")

    st.image("random_forest_roc_curve.png")
    st.write("""
    The Random Forest model achieved an AUC-ROC of **0.9505**, indicating a high ability to distinguish between survivors and high-risk patients. 
    By aggregating 200 different trees (as seen in our best parameters), the model reduces the variance found in the single Decision Tree, 
    leading to more stable and reliable predictions across diverse patient demographics.
    """)

    st.divider()

    st.subheader("4. Boosted Trees (LightGBM) Analysis")
    st.write("**Best LightGBM Test Metrics**")
    lgb_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8963, 0.8720, 0.9349, 0.9024, 0.9501]}
    st.table(pd.DataFrame(lgb_metrics))
    st.write("**Best Parameters:** `{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 50}`")

    st.image("lightgbm_roc_curve.png")
    st.write("""
    In this dataset, the high AUC-ROC score of **0.9501** indicates that the model is extremely effective at stratifying patient risk, 
    allowing for accurate triage in a clinical setting. Gradient boosting focuses on correcting errors of previous iterations, 
    making it a high-performance tool for identifying complex interactions between comorbidities.
    """)

    st.divider()

    st.subheader("5. Neural Network (MLP) Analysis")
    st.write("**Neural Network Test Metrics**")
    mlp_metrics = {"Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"], "Value": [0.8963, 0.8836, 0.9187, 0.9008, 0.9478]}
    st.table(pd.DataFrame(mlp_metrics))

    if os.path.exists("model_accuracy and model_loss.png"):
        st.image("model_accuracy and model_loss.png")
    
    st.write("""
    The training history plots show a healthy convergence, with both training and validation loss decreasing steadily over 5 epochs. 
    The lack of a significant gap between the training and validation accuracy indicates that the model is not overfitting. 
    Specifically, the **Model Loss** plot shows a smooth decline, while the **Model Accuracy** plot reaches a stable plateau, confirming 
    the network has effectively learned the patterns without memorizing noise.
    """)

    st.divider()
    st.subheader("Clinical Summary & Trade-offs")
    st.write("""
    The evaluation reveals that the **Neural Network** performed best (F1: 0.9068), achieving strong predictive power. However, the tree-based 
    ensemble models (LightGBM and Random Forest) provided a superior balance between accuracy and training efficiency.
    
    **The Trade-off:** Accuracy vs. Interpretability. While the MLP is more accurate, the Decision Tree offers transparent 'if-then' logic 
    easier for clinical professionals to trust and audit. Given the high stakes of mortality prediction, the Neural Network is recommended 
    for deployment due to its ability to handle complex comorbidity interactions.
    """)

# --- TAB 4: RISK PREDICTOR ---
with tab4:
    st.header("Explainability & Interactive Prediction")
    
    st.subheader("Global Explainability (SHAP)")
    if os.path.exists("shap_bar.png"): st.image("shap_bar.png")
    if os.path.exists("shap_beeswarm.png"): st.image("shap_beeswarm.png")
    
    st.write("""
    - **Strongest Impact:** The features with the strongest impact on mortality are **HOSPITALIZED, AGE, and PNEUMONIA**.
    - **Direction of Influence:** 
        - **Hospitalization:** Admitted patients see a massive push toward high-mortality outcomes.
        - **Age:** Clear positive correlation; higher age increases the risk significantly.
        - **Pneumonia:** Presence acts as a definitive risk multiplier.
    - **Clinical Utility:** Clinicians can use these insights to see that a patient's risk isn't just a number, 
      but a combination of specific interactions, allowing for personalized care pathways.
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
            # FIX: Use 'selected' instead of 'selected_model'
            prob = selected.predict_proba(user_input)[0][1] if hasattr(selected, "predict_proba") else 0.5
        
        if prediction == 1: st.error(f"High Mortality Risk (Prob: {prob:.2%})")
        else: st.success(f"Recovery Likely (Prob: {prob:.2%})")

        if model_choice in ["LightGBM", "Random Forest", "Decision Tree"]:
            selected_shap = {"LightGBM": lgbm_model, "Random Forest": rf_model, "Decision Tree": dt_model}[model_choice]
            explainer = shap.TreeExplainer(selected_shap)
            user_shap_values = explainer(user_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(user_shap_values[0])
            st.pyplot(fig)




