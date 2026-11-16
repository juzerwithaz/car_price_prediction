"""
AutoPrice Compass - Streamlit Web Application
An interactive valuation and market intelligence dashboard powered by 4 Machine Learning pillars.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.inference import (
    predict_car_valuation,
    get_feature_options,
    store,
    OUTPUTS_DIR
)
from src.utils import format_inr, format_lakhs, evaluate_deal

# Page configuration
st.set_page_config(
    page_title="AutoPrice Compass | Used Car Valuation",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern card UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .price-headline {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin: 5px 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-budget { background-color: #DEF7EC; color: #03543F; }
    .badge-mid { background-color: #E1EFFE; color: #1E429F; }
    .badge-premium { background-color: #F3E8FF; color: #6B21A8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car--v1.png", width=70)
    st.title("🧭 Compass Control")
    st.caption("AI-Powered Pre-owned Car Valuation")
    
    st.divider()
    st.subheader("⚙️ Model Architecture")
    reg_choice = st.selectbox(
        "Regression Model (Price)",
        ["Linear Regression (Best RMSE)", "Random Forest Regression"],
        index=0,
        help="Linear Regression achieved the lowest RMSE (~₹1.05 Lakh) on test data."
    )
    clf_choice = st.selectbox(
        "Classification Model (Tier)",
        ["Logistic Regression (Best F1)", "Random Forest Classification"],
        index=0,
        help="Logistic Regression achieved ~79.9% weighted F1 on tertile tiers."
    )
    
    st.divider()
    st.subheader("📊 Dataset Overview")
    st.markdown("""
    * **Source:** Cars24 Indian Market
    * **Listings Analyzed:** 8,015 vehicles
    * **Car Models:** 126 unique models
    * **Year Range:** 2010 – 2023
    """)
    st.info("💡 **Tip:** All models use zero target leakage and verified scikit-learn pipelines.")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🧭 AutoPrice Compass</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate Used-Car Valuation & Market Segmentation Engine</div>', unsafe_allow_html=True)

# Tabs
tab_val, tab_analytics, tab_explain = st.tabs([
    "🚗 Car Valuation & Deal Advisor",
    "📈 Market Analytics & ML Review",
    "💡 How It Works (Methodology)"
])

# Load feature options
options = get_feature_options()

# -----------------------------------------------------------------------------
# TAB 1: VALUATION & DEAL ADVISOR
# -----------------------------------------------------------------------------
with tab_val:
    col_input, col_result = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown("### 📝 Enter Car Details")
        st.caption("Provide the vehicle specifications to get an instant appraisal.")

        with st.form("car_valuation_form"):
            car_name = st.selectbox("Brand & Model", options["car_names"], index=0)
            
            f1, f2 = st.columns(2)
            with f1:
                year = st.selectbox("Manufacturing Year", options["years"], index=4) # 2019
            with f2:
                car_type = st.selectbox("Body Type", options["types"], index=0)

            f3, f4 = st.columns(2)
            with f3:
                fuel = st.selectbox("Fuel Type", options["fuels"], index=options["fuels"].index("PETROL") if "PETROL" in options["fuels"] else 0)
            with f4:
                drive = st.selectbox("Transmission", options["drives"], index=0)

            distance = st.number_input("Kilometers Driven", min_value=100, max_value=500000, value=42000, step=1000)
            
            f5, f6 = st.columns(2)
            with f5:
                owner = st.selectbox("Previous Owners", [1, 2, 3, 4], index=0)
            with f6:
                location = st.selectbox("Registration / Location", options["locations"], index=options["locations"].index("MH-14") if "MH-14" in options["locations"] else 0)

            submit_val = st.form_submit_button("⚡ Calculate Fair Market Price", use_container_width=True, type="primary")

    # Evaluate prediction
    res = predict_car_valuation(
        car_name=car_name,
        year=year,
        distance=distance,
        owner=owner,
        fuel=fuel,
        location=location,
        drive=drive,
        car_type=car_type,
        reg_model_name=reg_choice,
        clf_model_name=clf_choice
    )

    with col_result:
        st.markdown("### 🎯 Valuation Result")
        
        # Primary Price Card
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Estimated Fair Market Price</span>
            <div class="price-headline">{format_inr(res['fair_price'])}</div>
            <div style="color: #475569; font-size: 0.95rem;">
                <b>Expected Fair Range:</b> {format_lakhs(res['price_range_low'])} – {format_lakhs(res['price_range_high'])} 
                <span style="color: #94A3B8;">(± MAE: {format_lakhs(res['mae'])})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Segment & Regime Badges
        c1, c2 = st.columns(2)
        with c1:
            tier = res['tier']
            tier_class = "badge-mid"
            if tier == "Budget":
                tier_class = "badge-budget"
            elif tier == "Premium":
                tier_class = "badge-premium"

            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">PRICE CLASSIFICATION TIER</span>
                <div style="margin-top: 8px;">
                    <span class="badge-pill {tier_class}">{tier} Tier</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">
                    Identified via multi-class classifier without price leakage.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">K-MEANS MARKET CLUSTER</span>
                <div style="margin-top: 8px; font-weight: 700; color: #1E293B; font-size: 1.05rem;">
                    {res['cluster_name']}
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">
                    Cluster median price: {format_inr(res['cluster_stats'].get('median_price', 0))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # HMM Cohort Regime Box
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">HMM VALUE REGIME (COHORT ANALYSIS)</span>
            <div style="margin-top: 6px; font-size: 1.1rem; font-weight: 700; color: #0369A1;">
                🔄 {res['hmm_regime']}
            </div>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                Observed benchmark for <b>{year} {car_type}</b> cohorts: Median listed price is 
                <b>{format_inr(res['cohort_info'].get('cohort_median_price', res['fair_price']))}</b> with 
                average odometer reading of <b>{res['cohort_info'].get('cohort_median_distance', distance):,.0f} km</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Deal Meter
        st.divider()
        st.markdown("#### ⚖️ Deal Meter: Is your seller's price fair?")
        asking_input = st.number_input(
            "Enter Seller's Quoted Price (₹ INR)",
            min_value=0,
            value=int(round(res['fair_price'] * 1.04)),
            step=10000,
            help="Type what the dealer or seller is asking to see if it's a good deal."
        )

        deal = evaluate_deal(predicted_price=res['fair_price'], asking_price=asking_input)
        
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.metric(
                label="Asking vs Fair Value",
                value=f"{deal['percent']:+.1f}%",
                delta=f"{deal['difference']:+,.0f} INR",
                delta_color="inverse"
            )
        with d_col2:
            if deal['verdict'] == "success":
                st.success(f"**{deal['status']}**\n\n{deal['message']}")
            elif deal['verdict'] == "info":
                st.info(f"**{deal['status']}**\n\n{deal['message']}")
            elif deal['verdict'] == "warning":
                st.warning(f"**{deal['status']}**\n\n{deal['message']}")
            else:
                st.error(f"**{deal['status']}**\n\n{deal['message']}")

# -----------------------------------------------------------------------------
# TAB 2: MARKET ANALYTICS & ML REVIEWS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Comprehensive Model Evaluation & Analytics")
    st.caption("Real outputs and charts produced by the AutoPrice Compass pipeline.")

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "1. Regression (Continuous Price)",
        "2. Classification (Price Tiers)",
        "3. K-Means (Market Segments)",
        "4. Hidden Markov Model (Regimes)"
    ])

    with subtab1:
        st.subheader("Review 1: Regression Performance")
        reg_csv = OUTPUTS_DIR / "regression_comparison.csv"
        if reg_csv.exists():
            df_reg = pd.read_csv(reg_csv)
            st.dataframe(df_reg.style.format({
                "MAE_Rupees": "₹{:,.2f}",
                "RMSE_Rupees": "₹{:,.2f}",
                "R2_Score": "{:.4f}"
            }), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            p1 = OUTPUTS_DIR / "regression_actual_vs_predicted.png"
            if p1.exists():
                st.image(str(p1), caption="Actual vs Predicted Prices", use_container_width=True)
        with c2:
            p2 = OUTPUTS_DIR / "regression_model_comparison.png"
            if p2.exists():
                st.image(str(p2), caption="RMSE Comparison (Lower is better)", use_container_width=True)

    with subtab2:
        st.subheader("Review 1: Classification Performance")
        clf_csv = OUTPUTS_DIR / "classification_comparison.csv"
        if clf_csv.exists():
            df_clf = pd.read_csv(clf_csv)
            st.dataframe(df_clf.style.format({
                "Accuracy": "{:.4f}",
                "Precision_Weighted": "{:.4f}",
                "Recall_Weighted": "{:.4f}",
                "F1_Weighted": "{:.4f}"
            }), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            p1 = OUTPUTS_DIR / "classification_confusion_matrix.png"
            if p1.exists():
                st.image(str(p1), caption="Confusion Matrix (Logistic Regression)", use_container_width=True)
        with c2:
            p2 = OUTPUTS_DIR / "classification_model_comparison.png"
            if p2.exists():
                st.image(str(p2), caption="Weighted F1 Score Comparison", use_container_width=True)

    with subtab3:
        st.subheader("Review 2: K-Means Clustering Analysis")
        c_summary = OUTPUTS_DIR / "cluster_summary.csv"
        if c_summary.exists():
            st.markdown("**Cluster Summary Table:**")
            st.dataframe(pd.read_csv(c_summary), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            p1 = OUTPUTS_DIR / "kmeans_elbow.png"
            if p1.exists():
                st.image(str(p1), caption="Elbow Curve (Optimal K Selection)", use_container_width=True)
        with c2:
            p2 = OUTPUTS_DIR / "kmeans_clusters_pca.png"
            if p2.exists():
                st.image(str(p2), caption="PCA 2D Cluster Separation", use_container_width=True)

    with subtab4:
        st.subheader("Review 2: Gaussian Hidden Markov Model")
        hmm_summary = OUTPUTS_DIR / "hmm_state_summary.csv"
        if hmm_summary.exists():
            st.markdown("**Inferred Hidden State Characteristics:**")
            st.dataframe(pd.read_csv(hmm_summary), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            p1 = OUTPUTS_DIR / "hmm_vehicle_value_regimes.png"
            if p1.exists():
                st.image(str(p1), caption="Hidden Value Regimes Across Cohorts", use_container_width=True)
        with c2:
            p2 = OUTPUTS_DIR / "hmm_transition_matrix.png"
            if p2.exists():
                st.image(str(p2), caption="Regime Transition Probability Heatmap", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: HOW IT WORKS (METHODOLOGY)
# -----------------------------------------------------------------------------
with tab_explain:
    st.markdown("### 💡 Understanding the Machine Learning Architecture")
    
    st.markdown("""
    #### 1. Why Linear Regression Outperformed Random Forest
    When encoding high-cardinality categorical variables (such as 126 car models and dozens of RTO locations), One-Hot Encoding produces a sparse feature space. **Linear Regression** handles linear combinations of sparse binary flags with high stability. In contrast, standard un-tuned decision tree splits can struggle to generalize across rare categories without overfitting.

    #### 2. Prevention of Target Leakage
    In the **Classification** module, the target variable `Price_Category` was derived from tertiles of the vehicle price. To ensure rigorous evaluation without leakage, the numerical `Price` attribute is **strictly excluded** from the input feature set $X$. The classifier makes its predictions purely from vehicle specs, age, mileage, transmission, and location.

    #### 3. Dimension Prudence in Clustering
    K-Means clustering relies on Euclidean distance. If high-cardinality one-hot columns (`Car Name` and `Location`) were fed directly into K-Means, the distance metric would be dominated by orthogonal binary flags rather than meaningful physical parameters. We therefore clustered on scaled **Year, Log-Distance, Owner, Log-Price, Fuel, Transmission, and Body Type**.

    #### 4. Honest Framing of the Hidden Markov Model (HMM)
    The dataset's `Year` column represents the **vehicle manufacturing year**, not historical transaction timestamps. Therefore, the HMM is configured as an **exploratory cohort degradation model** grouped by `(Body Type, Year)`. It models how unobserved valuation regimes degrade as vehicle cohorts mature, rather than claiming to forecast individual stock-like price ticks.
    """)

# Footer
st.divider()
st.caption("AutoPrice Compass © 2026 | Built with Python, Streamlit, and Scikit-Learn | Academic Review 1 & 2 Implementation")
