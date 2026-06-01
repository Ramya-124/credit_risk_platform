import streamlit as st
import pandas as pd
import sqlite3
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="AI Credit Risk Platform",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------
# TITLE
# -----------------------------------------

# -----------------------------------------
# SIDEBAR
# -----------------------------------------

page = st.sidebar.selectbox(
    "Navigation",
    [
        "EDA & Business Insights",
        "Risk Prediction",
        "Explainability",
        "Business Rules",
        "Talk To Data"
    ]
)

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if page == "EDA & Business Insights":

    st.header("📈 EDA & Business Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "71.46%")

    with col2:
        st.metric("Recall", "67.17%")

    with col3:
        st.metric("ROC-AUC", "0.761")

    st.divider()

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Records", "307,511")

    with c2:
        st.metric("Features", "122")

    with c3:
        st.metric("Default Rate", "8.07%")

    st.divider()

    st.subheader("EDA Visualizations")

    st.image(
        "outputs/target_distribution.png",
        caption="Loan Default Distribution"
    )

    st.image(
        "outputs/correlation_heatmap.png",
        caption="Correlation Heatmap"
    )

    st.divider()
elif page == "Risk Prediction":

    st.header("⚠️ Credit Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=150000.0
        )

        credit = st.number_input(
            "Credit Amount",
            min_value=0.0,
            value=300000.0
        )

        annuity = st.number_input(
            "Loan Annuity",
            min_value=0.0,
            value=25000.0
        )

        goods_price = st.number_input(
            "Goods Price",
            min_value=0.0,
            value=200000.0
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        years_employed = st.number_input(
            "Years Employed",
            min_value=0,
            max_value=50,
            value=5
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"]
        )

        own_car = st.selectbox(
            "Own Car",
            ["Y", "N"]
        )

        own_realty = st.selectbox(
            "Own Property",
            ["Y", "N"]
        )

        education = st.selectbox(
            "Education",
            [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher"
            ]
        )

        family_status = st.selectbox(
            "Family Status",
            [
                "Married",
                "Single / not married",
                "Separated"
            ]
        )

        ext1 = st.slider(
    "External Score 1",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

        ext2 = st.slider(
    "External Score 2",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

        ext3 = st.slider(
    "External Score 3",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

    if st.button("Predict Risk"):

        try:

            model = joblib.load(
                "models/credit_risk_model.pkl"
            )

            encoders = joblib.load(
                "models/label_encoders.pkl"
            )

            medians = joblib.load(
                "models/median_values.pkl"
            )

            features = joblib.load(
                "models/feature_columns.pkl"
            )

            row = pd.DataFrame(
                index=[0],
                columns=features
            )

            for col in features:

                if col in medians:
                    row[col] = medians[col]
                else:
                    row[col] = 0

            if "AMT_INCOME_TOTAL" in row.columns:
                row["AMT_INCOME_TOTAL"] = income

            if "AMT_CREDIT" in row.columns:
                row["AMT_CREDIT"] = credit

            if "AMT_ANNUITY" in row.columns:
                row["AMT_ANNUITY"] = annuity

            if "AMT_GOODS_PRICE" in row.columns:
                row["AMT_GOODS_PRICE"] = goods_price

            if "DAYS_BIRTH" in row.columns:
                row["DAYS_BIRTH"] = -(age * 365)

            if "DAYS_EMPLOYED" in row.columns:
                row["DAYS_EMPLOYED"] = -(years_employed * 365)

            if "EXT_SOURCE_1" in row.columns:
                row["EXT_SOURCE_1"] = ext1

            if "EXT_SOURCE_2" in row.columns:
                row["EXT_SOURCE_2"] = ext2

            if "EXT_SOURCE_3" in row.columns:
                row["EXT_SOURCE_3"] = ext3

            if "CODE_GENDER" in row.columns:

                encoder = encoders["CODE_GENDER"]

                if gender in encoder.classes_:
                    row["CODE_GENDER"] = (
                        encoder.transform([gender])[0]
                    )

            if "FLAG_OWN_CAR" in row.columns:

                encoder = encoders["FLAG_OWN_CAR"]

                if own_car in encoder.classes_:
                    row["FLAG_OWN_CAR"] = (
                        encoder.transform([own_car])[0]
                    )

            if "FLAG_OWN_REALTY" in row.columns:

                encoder = encoders["FLAG_OWN_REALTY"]

                if own_realty in encoder.classes_:
                    row["FLAG_OWN_REALTY"] = (
                        encoder.transform([own_realty])[0]
                    )

            if "NAME_EDUCATION_TYPE" in row.columns:

                encoder = encoders["NAME_EDUCATION_TYPE"]

                if education in encoder.classes_:
                    row["NAME_EDUCATION_TYPE"] = (
                        encoder.transform([education])[0]
                    )

            if "NAME_FAMILY_STATUS" in row.columns:

                encoder = encoders["NAME_FAMILY_STATUS"]

                if family_status in encoder.classes_:
                    row["NAME_FAMILY_STATUS"] = (
                        encoder.transform([family_status])[0]
                    )
            st.write( row[[ "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AMT_CREDIT" ]] )
            probability = model.predict_proba(row)[0][1]

            risk_score = probability * 100

            st.subheader("Prediction Result")

            st.metric(
                "Default Probability",
                f"{risk_score:.2f}%"
            )

            if probability >= 0.70:
                st.error(
                    f"HIGH RISK ({risk_score:.2f}%)"
                )

            elif probability >= 0.40:
                st.warning(
                    f"MEDIUM RISK ({risk_score:.2f}%)"
                )

            else:
                st.success(
                    f"LOW RISK ({risk_score:.2f}%)"
                )

        except Exception as e:

            st.error(str(e))
# -----------------------------------------
# EXPLAINABILITY
# -----------------------------------------

elif page == "Explainability":

    st.header("🔍 SHAP Explainability")

    st.write("""
    SHAP (SHapley Additive Explanations)
    helps explain which features influence
    credit risk predictions the most.
    """)

    try:

        st.image(
            "outputs/shap_summary.png",
            caption="SHAP Feature Importance"
        )

    except:

        st.warning(
            "SHAP summary image not found."
        )

# -----------------------------------------
# BUSINESS RULES
# -----------------------------------------

elif page == "Business Rules":

    st.header("📋 Business Rules")

    st.code("""
RULE 1

IF Income < 100000
AND Credit > 500000

THEN HIGH RISK
""")

    st.code("""
RULE 2

IF Income between
100000 and 300000

THEN MEDIUM RISK
""")

    st.code("""
RULE 3

IF Income > 300000

THEN LOW RISK
""")

# -----------------------------------------
# TALK TO DATA
# -----------------------------------------

elif page == "Talk To Data":

    st.header("💬 Talk To Data")

    load_dotenv()

    question = st.text_input(
        "Ask a business question"
    )

    if st.button("Generate Answer"):

        if question:

            try:

                genai.configure(
                    api_key=os.getenv("GEMINI_API_KEY")
                )

                model = genai.GenerativeModel(
                    "models/gemini-3.5-flash"
                )

                df = pd.read_csv(
                    "data/application_train.csv"
                )

                columns = ", ".join(
                    df.columns.tolist()
                )

                prompt = f"""
                You are a SQL expert.

                Table Name: applications

                Available Columns:
                {columns}

                TARGET = 1 means default.
                TARGET = 0 means non-default.

                Generate ONLY SQLite SQL.

                Question:
                {question}
                """

                response = model.generate_content(
                    prompt
                )

                sql_query = (
                    response.text
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )

                st.subheader(
                    "Generated SQL"
                )

                st.code(sql_query)

                conn = sqlite3.connect(
                    "models/credit_risk.db"
                )

                result = pd.read_sql(
                    sql_query,
                    conn
                )

                st.subheader(
                    "Query Result"
                )

                st.dataframe(result)

                conn.close()

            except Exception as e:

                st.error(str(e))