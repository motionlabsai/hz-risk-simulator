import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HZ Risk Simulator",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #1a1a2e;
    padding: 1rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 500;
    color: #1a1a2e;
    font-family: 'DM Serif Display', serif;
}
.metric-label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.8rem 1rem;
    border-radius: 4px;
    margin: 0.5rem 0;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a1a2e;
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
}
.audience-tag {
    display: inline-block;
    background: #1a1a2e;
    color: white;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🏢 Hamilton Zanze")
st.markdown("### Monte Carlo Portfolio Risk Simulator")
st.markdown("*Built by Joshua Khurin — Summer 2025 Internship*")
st.divider()

# ── SIDEBAR — PROPERTY INPUTS ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Deal Inputs")
    st.markdown("*Adjust assumptions to model any property*")
    st.divider()

    property_name     = st.text_input("Property Name", "Sample HZ Value-Add Acquisition")
    purchase_price    = st.number_input("Purchase Price ($M)", min_value=1.0, max_value=500.0, value=65.0, step=1.0) * 1_000_000
    gross_revenue     = st.number_input("Gross Annual Revenue ($M)", min_value=0.1, max_value=50.0, value=6.0, step=0.1) * 1_000_000
    expense_ratio     = st.slider("Expense Ratio", min_value=0.30, max_value=0.65, value=0.42, step=0.01)
    hold_years        = st.slider("Hold Period (years)", min_value=3, max_value=10, value=5)
    ltv               = st.slider("LTV", min_value=0.40, max_value=0.80, value=0.60, step=0.01)
    equity_invested   = purchase_price * (1 - ltv)
    loan_amount       = purchase_price * ltv
    interest_rate     = st.slider("Interest Rate (%)", min_value=3.0, max_value=9.0, value=6.0, step=0.1) / 100
    annual_debt_service = loan_amount * interest_rate
    exit_cap_sigma    = st.slider("Exit Cap Rate Uncertainty (σ)", min_value=0.002, max_value=0.020, value=0.006, step=0.001)
    hurdle_rate       = st.slider("Hurdle Rate (equity multiple)", min_value=1.2, max_value=3.0, value=1.8, step=0.1)
    n_sims            = st.select_slider("Simulations", options=[1000, 5000, 10000], value=5000)

    st.divider()
    st.markdown(f"**Equity Invested:** ${equity_invested/1e6:.1f}M")
    st.markdown(f"**Loan Amount:** ${loan_amount/1e6:.1f}M")
    st.markdown(f"**Annual Debt Service:** ${annual_debt_service/1e6:.2f}M")

    noi_estimate = gross_revenue * (1 - expense_ratio)
    dscr_check   = noi_estimate / annual_debt_service if annual_debt_service > 0 else 0
    if dscr_check < 1.25:
        st.markdown(f'<div class="warning-box">⚠️ Going-in DSCR: {dscr_check:.2f}x — below lender minimum of 1.25x</div>', unsafe_allow_html=True)
    else:
        st.success(f"Going-in DSCR: {dscr_check:.2f}x ✓")

    run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── LOAD MACRO DATA ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_macro_data(api_key):
    import fredapi
    fred = fredapi.Fred(api_key=api_key)
    treasury_series     = fred.get_series('DGS10').resample('Y').mean().dropna()
    unemployment_series = fred.get_series('UNRATE').resample('Y').mean().dropna()
    cpi_series          = fred.get_series('CPIAUCSL').pct_change(12).resample('Y').mean().dropna() * 100
    gdp_series          = fred.get_series('A191RL1Q225SBEA').resample('Y').mean().dropna()
    return treasury_series, unemployment_series, cpi_series, gdp_series

# Try loading FRED data — fall back to hardcoded if key missing
try:
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", "b189465b27aff824c0a26416f3a69a9e")
    treasury_series, unemployment_series, cpi_series, gdp_series = load_macro_data(FRED_API_KEY)
    current_treasury     = treasury_series.iloc[-1]
    current_unemployment = unemployment_series.iloc[-1]
    current_cpi          = cpi_series.iloc[-1]
    current_gdp          = gdp_series.iloc[-1]
    macro_df = pd.DataFrame({
        "treasury":     treasury_series,
        "unemployment": unemployment_series,
        "cpi_growth":   cpi_series,
        "gdp_growth":   gdp_series,
    })
    macro_df["year"] = macro_df.index.year
    fred_loaded = True
except Exception:
    current_treasury     = 4.30
    current_unemployment = 4.10
    current_cpi          = 3.20
    current_gdp          = 2.50
    fred_loaded = False
    st.warning("FRED data unavailable — using hardcoded macro values")

# ── MACRO CONDITIONS STRIP ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Live Macro Conditions</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("10yr Treasury", f"{current_treasury:.2f}%")
with col2:
    st.metric("Unemployment", f"{current_unemployment:.2f}%")
with col3:
    st.metric("CPI Inflation", f"{current_cpi:.2f}%")
with col4:
    st.metric("GDP Growth", f"{current_gdp:.2f}%")

RISK_PREMIUM         = 0.02
exit_cap_mu_adjusted = (current_treasury / 100) + RISK_PREMIUM
st.caption(f"Exit cap rate anchored to Treasury + 200bps risk premium = **{exit_cap_mu_adjusted*100:.2f}%**")

# ── RUN SIMULATION ────────────────────────────────────────────────────────────
if run_button or True:  # auto-run on load with defaults

    np.random.seed(42)

    # Historical data — replace with real HZ data
    n_years = 7
    historical = pd.DataFrame({
        "year":           range(2018, 2018 + n_years),
        "rent_growth":    np.random.normal(2.8, 2.5, n_years),
        "vacancy_rate":   np.random.normal(5.2, 1.8, n_years).clip(0, 30),
        "expense_growth": np.random.normal(3.5, 1.5, n_years),
    })

    # MLE
    rent_mu,    rent_sig    = stats.norm.fit(historical["rent_growth"])
    vacancy_mu, vacancy_sig = stats.norm.fit(historical["vacancy_rate"])
    expense_mu, expense_sig = stats.norm.fit(historical["expense_growth"])
    cov_matrix = historical[["rent_growth", "vacancy_rate", "expense_growth"]].cov().values

    # Macro regression
    if fred_loaded:
        merged = pd.merge(historical, macro_df, on="year", how="inner")
        X         = merged[["treasury", "unemployment", "cpi_growth", "gdp_growth"]]
        y_rent    = merged["rent_growth"]
        y_vacancy = merged["vacancy_rate"]
        y_expense = merged["expense_growth"]
        rent_model    = LinearRegression().fit(X, y_rent)
        vacancy_model = LinearRegression().fit(X, y_vacancy)
        expense_model = LinearRegression().fit(X, y_expense)
        current_macro       = np.array([[current_treasury, current_unemployment, current_cpi, current_gdp]])
        rent_mu_adjusted    = rent_model.predict(current_macro)[0]
        vacancy_mu_adjusted = max(0, vacancy_model.predict(current_macro)[0])
        expense_mu_adjusted = expense_model.predict(current_macro)[0]
    else:
        rent_mu_adjusted    = rent_mu
        vacancy_mu_adjusted = vacancy_mu
        expense_mu_adjusted = expense_mu

    mu_vector = [rent_mu_adjusted, vacancy_mu_adjusted, expense_mu_adjusted]

    # MAP
    prior_rent_mu    = 2.8
    prior_vacancy_mu = 5.2
    prior_expense_mu = 3.5
    prior_sigma      = 1.5

    def map_estimate(data, prior_mu, prior_sigma):
        n            = len(data)
        data_mean    = data.mean()
        data_var     = data.var()
        prior_var    = prior_sigma ** 2
        posterior_mu = (prior_mu / prior_var + n * data_mean / data_var) / (1 / prior_var + n / data_var)
        return posterior_mu

    rent_mu_map    = map_estimate(historical["rent_growth"],    prior_rent_mu,    prior_sigma)
    vacancy_mu_map = map_estimate(historical["vacancy_rate"],   prior_vacancy_mu, prior_sigma)
    expense_mu_map = map_estimate(historical["expense_growth"], prior_expense_mu, prior_sigma)

    rent_mu_map_adjusted    = rent_mu_adjusted    + (rent_mu_map    - rent_mu)
    vacancy_mu_map_adjusted = max(0, vacancy_mu_adjusted + (vacancy_mu_map - vacancy_mu))
    expense_mu_map_adjusted = expense_mu_adjusted + (expense_mu_map - expense_mu)
    mu_vector_map           = [rent_mu_map_adjusted, vacancy_mu_map_adjusted, expense_mu_map_adjusted]

    PROPERTY = {
        "name":               property_name,
        "purchase_price":     purchase_price,
        "gross_revenue":      gross_revenue,
        "expense_ratio":      expense_ratio,
        "hold_years":         hold_years,
        "exit_cap_mu":        exit_cap_mu_adjusted,
        "exit_cap_sigma":     exit_cap_sigma,
        "equity_invested":    equity_invested,
        "annual_debt_service": annual_debt_service,
        "loan_balance_pct":   ltv,
    }

    # Monte Carlo simulation with progress bar
    with st.spinner(f"Running {n_sims:,} simulations..."):
        equity_multiples     = []
        equity_multiples_map = []
        dscr_values          = []
        dscr_values_map      = []

        for sim_num in range(n_sims):
            for use_map in [False, True]:
                mv = mu_vector_map if use_map else mu_vector
                revenue    = PROPERTY["gross_revenue"]
                cash_flows = [-PROPERTY["equity_invested"]]

                for year in range(PROPERTY["hold_years"]):
                    samples = np.random.multivariate_normal(mv, cov_matrix)
                    rg  = samples[0] / 100
                    v   = max(0, samples[1]) / 100
                    eg  = samples[2] / 100
                    revenue  = revenue * (1 + rg) * (1 - v)
                    expenses = revenue * PROPERTY["expense_ratio"] * (1 + eg)
                    noi      = revenue - expenses
                    cash_flows.append(noi - PROPERTY["annual_debt_service"])

                if use_map:
                    dscr_values_map.append(noi / PROPERTY["annual_debt_service"])
                else:
                    dscr_values.append(noi / PROPERTY["annual_debt_service"])

                exit_cap       = max(0.03, np.random.normal(PROPERTY["exit_cap_mu"], PROPERTY["exit_cap_sigma"]))
                sale_price     = noi / exit_cap
                debt_left      = PROPERTY["purchase_price"] * PROPERTY["loan_balance_pct"]
                cash_flows[-1] = cash_flows[-1] + sale_price - debt_left

                total = sum(cash_flows)
                em_val = (total + PROPERTY["equity_invested"]) / PROPERTY["equity_invested"]

                if use_map:
                    equity_multiples_map.append(em_val)
                else:
                    equity_multiples.append(em_val)

        em          = np.array(equity_multiples)
        em_map      = np.array(equity_multiples_map)
        dscr_values = np.array(dscr_values)
        dscr_values_map = np.array(dscr_values_map)

    # Risk metrics
    expected = em.mean()
    median   = np.median(em)
    std      = em.std()
    p10      = np.percentile(em, 10)
    p90      = np.percentile(em, 90)
    p_loss   = (em < 1.0).mean()
    p_2x     = (em > 2.0).mean()
    p_hurdle = (em > hurdle_rate).mean()
    avg_dscr = dscr_values.mean()
    p_dscr_distress = (dscr_values < 1.25).mean() * 100
    ci_lo, ci_hi = stats.norm.interval(0.95, loc=expected, scale=std / np.sqrt(n_sims))

    expected_map = em_map.mean()
    p_loss_map   = (em_map < 1.0).mean()
    p_hurdle_map = (em_map > hurdle_rate).mean()
    p10_map      = np.percentile(em_map, 10)

    # ── SECTION 1: ACQUISITIONS ───────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="audience-tag">Acquisitions Team</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Deal Risk Analysis</div>', unsafe_allow_html=True)

    # Key metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Expected Return", f"{expected:.2f}x", delta=f"±{std:.2f}x")
    with c2:
        st.metric("P10 (Bad Case)", f"{p10:.2f}x")
    with c3:
        st.metric("P90 (Good Case)", f"{p90:.2f}x")
    with c4:
        st.metric("P(Loss)", f"{p_loss*100:.1f}%", delta_color="inverse")
    with c5:
        st.metric(f"P(>{hurdle_rate}x)", f"{p_hurdle*100:.1f}%")
    with c6:
        st.metric("95% CI", f"({ci_lo:.2f}x, {ci_hi:.2f}x)")

    # Return distribution chart
    x_range      = np.linspace(em.min(), em.max(), 300)
    normal_curve = stats.norm.pdf(x_range, expected, std)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=em, nbinsx=80, histnorm="probability density",
        name="Simulated outcomes",
        marker=dict(color="#4A90D9", opacity=0.7),
        hovertemplate="Equity Multiple: %{x:.2f}x<br>Density: %{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_curve, mode="lines",
        name="Normal approx (CLT)",
        line=dict(color="black", dash="dash", width=2),
        hovertemplate="EM: %{x:.2f}x<br>Normal PDF: %{y:.4f}<extra></extra>"
    ))
    loss_x = x_range[x_range <= 1.0]
    loss_y = stats.norm.pdf(loss_x, expected, std)
    fig.add_trace(go.Scatter(
        x=np.concatenate([loss_x, loss_x[::-1]]),
        y=np.concatenate([loss_y, np.zeros(len(loss_y))]),
        fill="toself", fillcolor="rgba(231,76,60,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"Loss zone ({p_loss*100:.1f}%)", hoverinfo="skip"
    ))
    for label, val, color in [
        (f"P10: {p10:.2f}x",          p10,      "#e74c3c"),
        (f"Expected: {expected:.2f}x", expected, "#1a1a2e"),
        (f"P90: {p90:.2f}x",          p90,      "#27ae60"),
        ("Breakeven",                  1.0,      "gray"),
    ]:
        fig.add_vline(x=val, line_color=color, line_dash="dash", line_width=1.8,
                      annotation_text=label, annotation_position="top", annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f"Distribution of Returns — {PROPERTY['name']}<br><sup>{n_sims:,} Monte Carlo simulations · {hold_years}-year hold</sup>", font=dict(size=15)),
        xaxis=dict(title="Equity Multiple (x)", tickformat=".1f", gridcolor="#f0f0f0"),
        yaxis=dict(title="Probability Density", gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=450,
        font=dict(family="DM Sans", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    # MLE vs MAP comparison
    st.markdown("**MLE vs MAP Comparison**")
    st.caption("MAP tempers estimates with CBRE market priors — use for new markets with limited data history")
    comparison_df = pd.DataFrame({
        "Metric":    ["Expected EM", "P10 (bad case)", "P(loss)", f"P(>{hurdle_rate}x hurdle)"],
        "MLE Model": [f"{expected:.2f}x", f"{p10:.2f}x", f"{p_loss*100:.1f}%", f"{p_hurdle*100:.1f}%"],
        "MAP Model": [f"{expected_map:.2f}x", f"{p10_map:.2f}x", f"{p_loss_map*100:.1f}%", f"{p_hurdle_map*100:.1f}%"],
    })
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # Sensitivity tornado chart
    st.markdown("**Sensitivity Analysis**")

    def quick_sim(r_mu, v_mu, e_mu, cov, n=1000):
        mu_vec = [r_mu, v_mu, e_mu]
        ems = []
        for _ in range(n):
            rev = PROPERTY["gross_revenue"]
            cfs = [-PROPERTY["equity_invested"]]
            for _ in range(PROPERTY["hold_years"]):
                s   = np.random.multivariate_normal(mu_vec, cov)
                rg  = s[0] / 100
                v   = max(0, s[1]) / 100
                eg  = s[2] / 100
                rev = rev * (1 + rg) * (1 - v)
                exp = rev * PROPERTY["expense_ratio"] * (1 + eg)
                noi = rev - exp
                cfs.append(noi - PROPERTY["annual_debt_service"])
            ec  = max(0.03, np.random.normal(PROPERTY["exit_cap_mu"], PROPERTY["exit_cap_sigma"]))
            sp  = noi / ec
            cfs[-1] = cfs[-1] + sp - PROPERTY["purchase_price"] * PROPERTY["loan_balance_pct"]
            total = sum(cfs)
            ems.append((total + PROPERTY["equity_invested"]) / PROPERTY["equity_invested"])
        return np.mean(ems)

    with st.spinner("Running sensitivity analysis..."):
        base             = quick_sim(rent_mu_adjusted,               vacancy_mu_adjusted,               expense_mu_adjusted,               cov_matrix)
        rent_growth_up   = quick_sim(rent_mu_adjusted + rent_sig,    vacancy_mu_adjusted,               expense_mu_adjusted,               cov_matrix) - base
        rent_growth_down = quick_sim(rent_mu_adjusted - rent_sig,    vacancy_mu_adjusted,               expense_mu_adjusted,               cov_matrix) - base
        vacancy_up       = quick_sim(rent_mu_adjusted,               vacancy_mu_adjusted + vacancy_sig, expense_mu_adjusted,               cov_matrix) - base
        vacancy_down     = quick_sim(rent_mu_adjusted,               vacancy_mu_adjusted - vacancy_sig, expense_mu_adjusted,               cov_matrix) - base
        expenses_up      = quick_sim(rent_mu_adjusted,               vacancy_mu_adjusted,               expense_mu_adjusted + expense_sig, cov_matrix) - base
        expenses_down    = quick_sim(rent_mu_adjusted,               vacancy_mu_adjusted,               expense_mu_adjusted - expense_sig, cov_matrix) - base

    labels = ["Rent +1σ", "Rent -1σ", "Vacancy +1σ", "Vacancy -1σ", "Expenses +1σ", "Expenses -1σ"]
    values = [rent_growth_up, rent_growth_down, vacancy_up, vacancy_down, expenses_up, expenses_down]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.barh(labels, values, color=colors, alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Change in Expected Equity Multiple (x)")
    ax2.set_title("Sensitivity Analysis — Which Variable Drives Risk Most?\n(longer bar = bigger impact, focus due diligence here)", fontsize=11)
    sns.despine(ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── SECTION 2: INVESTOR RELATIONS ─────────────────────────────────────────
    st.divider()
    st.markdown('<span class="audience-tag">Investor Relations & Capital Markets</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">LP Reporting & Lender Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Avg DSCR (final year)", f"{avg_dscr:.2f}x", delta="Lender min: 1.25x")
    with col_b:
        st.metric("P(DSCR Distress)", f"{p_dscr_distress:.1f}%",
                  delta="High risk" if p_dscr_distress > 15 else "Acceptable",
                  delta_color="inverse" if p_dscr_distress > 15 else "normal")

    # Exceedance curve
    thresholds      = np.linspace(0.5, 3.0, 200)
    exceedance_mle  = [(em > t).mean()     for t in thresholds]
    exceedance_map  = [(em_map > t).mean() for t in thresholds]
    hurdle_prob_mle = (em > hurdle_rate).mean()
    hurdle_prob_map = (em_map > hurdle_rate).mean()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(thresholds, exceedance_mle, color="steelblue", lw=2, label="MLE model")
    ax3.plot(thresholds, exceedance_map, color="orange",    lw=2, label="MAP model")
    ax3.fill_between(thresholds, exceedance_mle, exceedance_map, alpha=0.1, color="gray", label="Uncertainty band")
    ax3.axvline(hurdle_rate, color="green",     ls="--", lw=1.8, label=f"Hurdle: {hurdle_rate}x")
    ax3.axhline(hurdle_prob_mle, color="steelblue", ls=":", lw=1.2, label=f"MLE P(>{hurdle_rate}x) = {hurdle_prob_mle*100:.1f}%")
    ax3.axhline(hurdle_prob_map, color="orange",    ls=":", lw=1.2, label=f"MAP P(>{hurdle_rate}x) = {hurdle_prob_map*100:.1f}%")
    ax3.axvline(1.0, color="red", ls="--", lw=1.0, alpha=0.5, label="Breakeven")
    ax3.set_xlabel("Equity Multiple (x)")
    ax3.set_ylabel("Probability of Exceeding Threshold")
    ax3.set_title("Exceedance Probability — MLE vs MAP\nGap = uncertainty from limited data history", fontsize=11)
    ax3.legend(fontsize=8)
    sns.despine(ax=ax3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Stress test
    st.markdown("**Stress Test Scenarios**")
    with st.spinner("Running stress tests..."):
        scenarios = {
            "Base Case":          (rent_mu_adjusted,       vacancy_mu_adjusted,       expense_mu_adjusted),
            "Recession":          (rent_mu_adjusted - 2.0, vacancy_mu_adjusted + 3.0, expense_mu_adjusted + 1.5),
            "Rate Normalization": (rent_mu_adjusted + 1.5, vacancy_mu_adjusted - 1.0, expense_mu_adjusted),
        }
        stress_results = []
        for scenario_name, (r_mu, v_mu, e_mu) in scenarios.items():
            s_ems = []
            for _ in range(2000):
                rev = PROPERTY["gross_revenue"]
                cfs = [-PROPERTY["equity_invested"]]
                for _ in range(PROPERTY["hold_years"]):
                    s   = np.random.multivariate_normal([r_mu, v_mu, e_mu], cov_matrix)
                    rg  = s[0] / 100
                    v   = max(0, s[1]) / 100
                    eg  = s[2] / 100
                    rev = rev * (1 + rg) * (1 - v)
                    exp = rev * PROPERTY["expense_ratio"] * (1 + eg)
                    noi = rev - exp
                    cfs.append(noi - PROPERTY["annual_debt_service"])
                ec  = max(0.03, np.random.normal(PROPERTY["exit_cap_mu"], PROPERTY["exit_cap_sigma"]))
                sp  = noi / ec
                cfs[-1] = cfs[-1] + sp - PROPERTY["purchase_price"] * PROPERTY["loan_balance_pct"]
                total = sum(cfs)
                s_ems.append((total + PROPERTY["equity_invested"]) / PROPERTY["equity_invested"])
            s_ems = np.array(s_ems)
            stress_results.append({
                "Scenario":       scenario_name,
                "Expected EM":    f"{s_ems.mean():.2f}x",
                "P10 (bad case)": f"{np.percentile(s_ems, 10):.2f}x",
                "P(loss)":        f"{(s_ems < 1.0).mean() * 100:.1f}%",
                f"P(>{hurdle_rate}x)": f"{(s_ems > hurdle_rate).mean() * 100:.1f}%",
            })
    stress_df = pd.DataFrame(stress_results).set_index("Scenario")
    st.dataframe(stress_df, use_container_width=True)

    # ── SECTION 3: ASSET MANAGEMENT ───────────────────────────────────────────
    st.divider()
    st.markdown('<span class="audience-tag">Asset Management</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Portfolio Occupancy Monitoring</div>', unsafe_allow_html=True)
    st.caption("Replace transition matrix P with probabilities from real unit-level lease data")

    P = np.array([
        [0.92, 0.06, 0.02, 0.00],
        [0.05, 0.10, 0.85, 0.00],
        [0.00, 0.00, 0.30, 0.70],
        [0.75, 0.00, 0.05, 0.20],
    ])

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx        = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Long-run Occupancy", f"{stationary[0]*100:.1f}%",
                  delta="Below 90% target" if stationary[0] < 0.90 else "Above 90% target",
                  delta_color="inverse" if stationary[0] < 0.90 else "normal")
    with col_m2:
        states_df = pd.DataFrame({
            "State":       ["Occupied", "Notice Given", "Vacant", "Leasing"],
            "Steady State": [f"{p*100:.1f}%" for p in stationary]
        })
        st.dataframe(states_df, hide_index=True, use_container_width=True)

    N_UNITS     = 80
    N_MONTHS    = 24
    unit_states = np.zeros(N_UNITS, dtype=int)
    monthly_occ = []
    for _ in range(N_MONTHS):
        new_states = []
        for s in unit_states:
            new_states.append(np.random.choice(4, p=P[s]))
        unit_states = np.array(new_states)
        monthly_occ.append((unit_states == 0).mean() * 100)

    fig4, ax4 = plt.subplots(figsize=(11, 4))
    ax4.plot(range(1, N_MONTHS + 1), monthly_occ, color="steelblue", lw=2, label="Simulated occupancy")
    ax4.axhline(stationary[0] * 100, color="red", ls="--", lw=1.8, label=f"Steady state: {stationary[0]*100:.1f}%")
    ax4.fill_between(range(1, N_MONTHS + 1), 90, 102, alpha=0.07, color="green", label="Target zone (>90%)")
    ax4.set_ylim(60, 102)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Occupancy Rate (%)")
    ax4.set_title(f"Markov Chain Occupancy Forecast — {N_UNITS} Units over {N_MONTHS} Months", fontsize=11)
    ax4.legend(fontsize=9)
    sns.despine(ax=ax4)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── SECTION 4: MODEL METHODOLOGY ──────────────────────────────────────────
    with st.expander("📐 Model Methodology (Internal)", expanded=False):
        st.markdown('<span class="audience-tag">Internal</span>', unsafe_allow_html=True)
        st.markdown("**MLE vs MAP Parameter Estimates**")
        method_df = pd.DataFrame({
            "Variable":       ["Rent Growth", "Vacancy Rate", "Expense Growth"],
            "MLE (μ)":        [f"{rent_mu:.2f}%", f"{vacancy_mu:.2f}%", f"{expense_mu:.2f}%"],
            "MAP (μ)":        [f"{rent_mu_map:.2f}%", f"{vacancy_mu_map:.2f}%", f"{expense_mu_map:.2f}%"],
            "Prior (μ)":      [f"{prior_rent_mu:.2f}%", f"{prior_vacancy_mu:.2f}%", f"{prior_expense_mu:.2f}%"],
            "Macro-Adjusted": [f"{rent_mu_adjusted:.2f}%", f"{vacancy_mu_adjusted:.2f}%", f"{expense_mu_adjusted:.2f}%"],
        })
        st.dataframe(method_df, hide_index=True, use_container_width=True)

        st.markdown("**Correlation Matrix**")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            historical[["rent_growth", "vacancy_rate", "expense_growth"]].corr(),
            annot=True, cmap="coolwarm", center=0, ax=ax5, fmt=".2f"
        )
        ax5.set_title("Rent Growth / Vacancy / Expense Correlation")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    st.divider()
    st.caption("Hamilton Zanze Monte Carlo Portfolio Risk Simulator · Built by Joshua Khurin · Summer 2025")
