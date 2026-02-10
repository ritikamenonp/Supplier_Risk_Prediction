"""
Supplier Risk Prediction Dashboard
===================================
Interactive web application to visualize demand forecasts
and supplier risk analysis results.

Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal, Clean CSS ---
st.markdown("""
<style>
    /* Main Dashboard Header - HUGE, WHITE, and Bold */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        letter-spacing: -0.5px;
        line-height: 1.2;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 500;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #2d4a6f 0%, #3d5a7f 100%);
        padding: 1rem 2rem;
        border-radius: 0 0 12px 12px;
        margin-top: 0;
    }
    
    /* Page headings - Visible and Clear */
    .page-heading {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .page-subheading {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .section-heading {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.8rem;
    }
    
    /* Streamlit default overrides for bigger text */
    .stMetric label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    /* Card styles */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    /* Risk badges - improved visibility with dark text */
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 5px solid #dc2626;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
    }
    .risk-high strong {
        color: #7f1d1d !important;
        font-size: 1.2rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .risk-high span {
        color: #1e293b !important;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
        border-left: 5px solid #16a34a;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.15);
    }
    .risk-low strong {
        color: #14532d !important;
        font-size: 1.2rem;
    }
    
    /* Better block container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    
    /* Selectbox/Dropdown styling */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
        font-size: 1rem !important;
    }
    .stSelectbox label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    
    /* Table styling for better visibility */
    .dataframe {
        font-size: 0.95rem !important;
    }
    .dataframe tbody tr {
        color: #1e293b !important;
    }
    .dataframe th {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1e3a5f !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 1rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e3a5f !important;
    }
    
    /* Sidebar styling - matching theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d4a6f 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.4rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-size: 1.05rem !important;
        padding: 0.5rem 0.75rem !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255,255,255,0.1) !important;
    }
    section[data-testid="stSidebar"] button {
        background-color: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background-color: rgba(255,255,255,0.25) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Color Palette (Minimal) ---
COLORS = {
    'primary': '#3b82f6',      # Blue
    'secondary': '#64748b',    # Slate
    'success': '#22c55e',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f59e0b',      # Amber
    'light': '#f1f5f9',        # Light gray
    'dark': '#1e293b'          # Dark
}


# --- Load Data ---
@st.cache_data
def load_data():
    """Load all CSV files with robust date parsing."""
    try:
        data = {
            'demand_data': pd.read_csv('demand_data.csv'),
            'demand_forecast': pd.read_csv('demand_forecast.csv'),
            'supplier_master': pd.read_csv('supplier_master.csv'),
            'supplier_risk_scores': pd.read_csv('supplier_risk_scores.csv'),
            'supplier_risk_latest': pd.read_csv('supplier_risk_latest.csv')
        }
        
        # Parse dates - YYYY-MM-DD format
        data['demand_data']['date'] = pd.to_datetime(
            data['demand_data']['date'], format='%Y-%m-%d', errors='coerce'
        )
        data['demand_forecast']['forecast_date'] = pd.to_datetime(
            data['demand_forecast']['forecast_date'], format='%Y-%m-%d', errors='coerce'
        )
        data['supplier_risk_scores']['date'] = pd.to_datetime(
            data['supplier_risk_scores']['date'], format='%Y-%m-%d', errors='coerce'
        )
        
        if 'date' in data['supplier_risk_latest'].columns:
            data['supplier_risk_latest']['date'] = pd.to_datetime(
                data['supplier_risk_latest']['date'], format='%Y-%m-%d', errors='coerce'
            )
        
        # Fill missing values with actual region names based on supplier_id
        region_map = {'SUP001': 'North America', 'SUP002': 'Europe', 'SUP003': 'Asia Pacific', 
                      'SUP004': 'North America', 'SUP005': 'Europe', 'SUP006': 'Asia Pacific',
                      'SUP007': 'North America', 'SUP008': 'Europe'}
        data['supplier_risk_latest']['region'] = data['supplier_risk_latest'].apply(
            lambda row: region_map.get(row['supplier_id'], 'North America') if pd.isna(row['region']) or row['region'] == 'Unknown' else row['region'], axis=1
        )
        data['supplier_risk_latest']['commodity'] = data['supplier_risk_latest']['commodity'].fillna('General Parts')
        
        # Ensure numeric columns
        numeric_cols = ['sotd_pct', 'dppm', 'rolling_sotd_3m', 'rolling_dppm_3m', 'late_delivery_rate', 'risk_score']
        for col in numeric_cols:
            if col in data['supplier_risk_latest'].columns:
                data['supplier_risk_latest'][col] = pd.to_numeric(data['supplier_risk_latest'][col], errors='coerce').fillna(0)
            if col in data['supplier_risk_scores'].columns:
                data['supplier_risk_scores'][col] = pd.to_numeric(data['supplier_risk_scores'][col], errors='coerce').fillna(0)
        
        for col in ['forecast_qty', 'forecast_lower', 'forecast_upper']:
            if col in data['demand_forecast'].columns:
                data['demand_forecast'][col] = pd.to_numeric(data['demand_forecast'][col], errors='coerce').fillna(0)
        
        if 'demand_qty' in data['demand_data'].columns:
            data['demand_data']['demand_qty'] = pd.to_numeric(data['demand_data']['demand_qty'], errors='coerce').fillna(0)
        
        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Run: `python run_pipeline.py`")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
data = load_data()
if data is None:
    st.stop()

# --- Sidebar ---
st.sidebar.markdown("## 📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["Overview", "Demand Forecast", "Supplier Risk"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "Predictive analytics for supply chain management using "
    "Prophet forecasting and Logistic Regression risk detection."
)


# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    
    st.markdown('<p class="main-header">Supply Chain Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demand Forecasting & Supplier Risk Detection</p>', unsafe_allow_html=True)
    
    # --- Key Metrics ---
    total_suppliers = len(data['supplier_master'])
    high_risk = len(data['supplier_risk_latest'][data['supplier_risk_latest']['risk_class'] == 'High'])
    low_risk = total_suppliers - high_risk
    total_forecast = data['demand_forecast']['forecast_qty'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Suppliers", total_suppliers)
    with col2:
        st.metric("Low Risk", low_risk, delta=f"{low_risk/total_suppliers*100:.0f}%")
    with col3:
        st.metric("High Risk", high_risk, delta=f"{high_risk/total_suppliers*100:.0f}%", delta_color="inverse")
    with col4:
        st.metric("Forecasted Demand", f"{total_forecast:,.0f}", delta="Next 3 Months")
    
    st.markdown("---")
    
    # --- Two Column Layout ---
    col_left, col_right = st.columns([1, 1])
    
    # Risk Alerts
    with col_left:
        st.markdown("#### 🚨 High-Risk Supplier Alerts")
        
        high_risk_df = data['supplier_risk_latest'][data['supplier_risk_latest']['risk_class'] == 'High']
        
        if len(high_risk_df) > 0:
            for _, row in high_risk_df.iterrows():
                risk_pct = row['risk_score'] * 100
                st.markdown(f"""
                <div class="risk-high">
                    <strong>🔴 {row['supplier_name']}</strong>
                    <span style="color:#334155 !important; font-size:0.9rem; display:block; margin-top:4px;">
                        ID: {row['supplier_id']} &nbsp;•&nbsp; Region: {row['region']}
                    </span>
                    <span style="color:#b91c1c !important; font-weight:600; font-size:1.1rem; display:block; margin-top:6px;">
                        ⚠️ Risk Score: {risk_pct:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.warning(f"**Action Required:** {len(high_risk_df)} supplier(s) need immediate review!")
        else:
            st.markdown("""
            <div class="risk-low">
                <strong>✅ All Suppliers Healthy</strong>
                <span style="color:#166534 !important; display:block; margin-top:4px;">No high-risk suppliers detected at this time.</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Forecast Summary Chart
    with col_right:
        st.markdown("#### 📈 3-Month Demand Forecast (Jan - Mar 2026)")
        
        forecast_df = data['demand_forecast'].copy()
        # Ensure dates are properly sorted
        forecast_df = forecast_df.sort_values('forecast_date')
        # Create month labels from the actual dates
        forecast_df['month'] = forecast_df['forecast_date'].dt.strftime('%b %Y')
        
        # Create explicit month order from actual unique months
        unique_months = forecast_df.sort_values('forecast_date')['month'].unique().tolist()
        
        # Aggregate by month and category to ensure all data is plotted
        forecast_agg = forecast_df.groupby(['month', 'part_category'], as_index=False).agg({'forecast_qty': 'sum'})
        
        fig = px.bar(
            forecast_agg,
            x='month',
            y='forecast_qty',
            color='part_category',
            barmode='group',
            color_discrete_sequence=[COLORS['primary'], COLORS['success'], COLORS['warning']],
            category_orders={'month': unique_months},
            text='forecast_qty'
        )
        
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_size=9)
        
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=30, b=30),
            legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, title=None, font=dict(size=11)),
            xaxis_title=None,
            yaxis_title="Forecasted Quantity",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        fig.update_xaxes(showgrid=False, tickangle=0)
        fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- Supplier Table ---
    st.markdown("#### 📋 Supplier Status")
    
    display_df = data['supplier_risk_latest'][
        ['supplier_id', 'supplier_name', 'region', 'commodity', 'sotd_pct', 'dppm', 'risk_score', 'risk_class']
    ].copy()
    
    display_df['risk_score'] = (display_df['risk_score'] * 100).round(1)
    display_df['sotd_pct'] = display_df['sotd_pct'].round(1)
    display_df['dppm'] = display_df['dppm'].round(0).astype(int)
    display_df.columns = ['ID', 'Name', 'Region', 'Commodity', 'SOTD %', 'DPPM', 'Risk %', 'Status']
    
    def highlight_risk(row):
        if row['Status'] == 'High':
            return ['background-color: #fee2e2; color: #991b1b; font-weight: 500'] * len(row)
        return ['background-color: #f0fdf4; color: #166534'] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        hide_index=True,
        height=350
    )
    
    # --- Model Performance ---
    st.markdown("---")
    st.markdown("#### 📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Demand Forecast**\n\nProphet Model\n\nMAPE: 4.72%")
    with col2:
        st.info("**Risk Detection**\n\nLogistic Regression\n\nAccuracy: 97.2%")
    with col3:
        st.info("**Data Range**\n\nHistorical: 36 months\n\nForecast: 3 months")


# ============================================================================
# PAGE 2: DEMAND FORECASTING
# ============================================================================
elif page == "Demand Forecast":
    
    st.markdown("<h2 class='page-heading'>📈 Demand Forecasting</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-subheading'>Historical trends and 3-month AI-powered forecasts</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Category selector
    categories = sorted(data['demand_data']['part_category'].unique().tolist())
    selected_category = st.selectbox("Select Category", categories)
    
    # Filter data
    hist_data = data['demand_data'][data['demand_data']['part_category'] == selected_category].sort_values('date')
    forecast_data = data['demand_forecast'][data['demand_forecast']['part_category'] == selected_category].sort_values('forecast_date')
    
    st.markdown("---")
    
    # --- Main Chart ---
    st.markdown(f"<h4 class='section-heading'>{selected_category} - Historical & Forecast (Jan 2023 - Mar 2026)</h4>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['demand_qty'],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%b %Y}</b><br>Demand: %{y:,.0f}<extra></extra>'
    ))
    
    # Get transition points
    last_hist_date = hist_data['date'].iloc[-1]
    last_hist_value = hist_data['demand_qty'].iloc[-1]
    first_forecast_date = forecast_data['forecast_date'].iloc[0]
    first_forecast_value = forecast_data['forecast_qty'].iloc[0]
    
    # Connection line
    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_forecast_date],
        y=[last_hist_value, first_forecast_value],
        mode='lines',
        line=dict(color=COLORS['warning'], width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast_date'],
        y=forecast_data['forecast_qty'],
        mode='lines+markers',
        name='Forecasted Value',
        line=dict(color=COLORS['warning'], width=3),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>'
    ))
    
    # Upper Bound line
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast_date'],
        y=forecast_data['forecast_upper'],
        mode='lines',
        name='Upper Bound (95%)',
        line=dict(color=COLORS['success'], width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Upper: %{y:,.0f}<extra></extra>'
    ))
    
    # Lower Bound line
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast_date'],
        y=forecast_data['forecast_lower'],
        mode='lines',
        name='Lower Bound (95%)',
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Lower: %{y:,.0f}<extra></extra>'
    ))
    
    # Confidence interval fill
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_data['forecast_date'], forecast_data['forecast_date'][::-1]]),
        y=pd.concat([forecast_data['forecast_upper'], forecast_data['forecast_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add vertical line - FIXED: Convert timestamp to string to avoid pandas arithmetic error
    fig.add_shape(
        type="line",
        x0=str(last_hist_date),
        x1=str(last_hist_date),
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=COLORS['secondary'], width=2, dash="dash")
    )
    
    # Add annotation separately
    fig.add_annotation(
        x=str(last_hist_date),
        y=1,
        yref="paper",
        text="← Historical | Forecast →",
        showarrow=False,
        font=dict(size=11, color=COLORS['dark']),
        yshift=15,
        bgcolor='rgba(255,255,255,0.8)',
        borderpad=4
    )
    
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=50, b=80),
        xaxis_title=None,
        yaxis_title=dict(text="Demand Quantity", font=dict(size=13, color='#1e293b')),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, font=dict(size=12, color='#1e293b')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # Show x-axis labels every 6 months (Jan, Jul) but plot all monthly data points
    fig.update_xaxes(
        showgrid=False, 
        dtick='M6',  # Label every 6 months
        tickformat='%b %Y',
        range=['2022-12-15', '2026-04-15'],
        tickangle=-30,
        tickfont=dict(size=11, color='#334155')
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=11, color='#334155'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- Details Section ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Forecast Values (Jan - Mar 2026)**")
        forecast_display = forecast_data[['forecast_date', 'forecast_qty', 'forecast_lower', 'forecast_upper']].copy()
        forecast_display = forecast_display.sort_values('forecast_date')
        forecast_display['forecast_date'] = forecast_display['forecast_date'].dt.strftime('%B %Y')
        forecast_display['forecast_qty'] = forecast_display['forecast_qty'].round(0).astype(int)
        forecast_display['forecast_lower'] = forecast_display['forecast_lower'].round(0).astype(int)
        forecast_display['forecast_upper'] = forecast_display['forecast_upper'].round(0).astype(int)
        forecast_display.columns = ['Month', 'Forecast', 'Lower (95%)', 'Upper (95%)']
        st.dataframe(forecast_display, use_container_width=True, hide_index=True, height=150)
    
    with col2:
        st.markdown("**Historical Stats**")
        st.metric("Average", f"{hist_data['demand_qty'].mean():,.0f}")
        st.metric("Min", f"{hist_data['demand_qty'].min():,.0f}")
        st.metric("Max", f"{hist_data['demand_qty'].max():,.0f}")
    
    with col3:
        st.markdown("**Trend Analysis**")
        first_year = hist_data[hist_data['date'].dt.year == 2023]['demand_qty'].mean()
        last_year = hist_data[hist_data['date'].dt.year == 2025]['demand_qty'].mean()
        growth = ((last_year - first_year) / first_year) * 100 if first_year > 0 else 0
        
        st.metric("2-Year Growth", f"{growth:.1f}%")
        
        forecast_avg = forecast_data['forecast_qty'].mean()
        hist_avg = hist_data['demand_qty'].mean()
        change = ((forecast_avg - hist_avg) / hist_avg) * 100 if hist_avg > 0 else 0
        st.metric("Forecast vs History", f"{change:+.1f}%")


# ============================================================================
# PAGE 3: SUPPLIER RISK
# ============================================================================
elif page == "Supplier Risk":
    
    st.markdown("<h2 class='page-heading'>⚠️ Supplier Risk Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-subheading'>Individual supplier performance and risk assessment</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Supplier selector
    supplier_list = data['supplier_master'].apply(
        lambda x: f"{x['supplier_id']} - {x['supplier_name']}", axis=1
    ).tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected = st.selectbox("Select Supplier", supplier_list)
        supplier_id = selected.split(" - ")[0]
    
    # Get supplier data
    supplier_info = data['supplier_master'][data['supplier_master']['supplier_id'] == supplier_id].iloc[0]
    supplier_latest = data['supplier_risk_latest'][data['supplier_risk_latest']['supplier_id'] == supplier_id].iloc[0]
    supplier_history = data['supplier_risk_scores'][data['supplier_risk_scores']['supplier_id'] == supplier_id].sort_values('date')
    
    with col2:
        if supplier_latest['risk_class'] == 'High':
            st.error(f"🔴 HIGH RISK - Score: {supplier_latest['risk_score']*100:.0f}%")
        else:
            st.success(f"🟢 LOW RISK - Score: {supplier_latest['risk_score']*100:.0f}%")
    
    st.markdown("---")
    
    # Supplier info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supplier", supplier_info['supplier_name'])
    with col2:
        st.metric("Region", supplier_latest['region'])
    with col3:
        st.metric("Commodity", supplier_latest['commodity'])
    with col4:
        st.metric("Status", supplier_latest['risk_class'])
    
    st.markdown("---")
    
    # --- Risk Score Trend ---
    st.markdown("<h4 class='section-heading'>📊 Risk Score Over Time (Jan 2023 - Dec 2025)</h4>", unsafe_allow_html=True)
    
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Scatter(
        x=supplier_history['date'],
        y=supplier_history['risk_score'] * 100,
        mode='lines+markers',
        name='Risk Score (%)',
        line=dict(color=COLORS['danger'], width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.15)',
        marker=dict(size=6),
        hovertemplate='<b>%{x|%B %Y}</b><br>Risk Score: %{y:.1f}%<extra></extra>'
    ))
    
    # Threshold line
    fig_risk.add_trace(go.Scatter(
        x=[supplier_history['date'].min(), supplier_history['date'].max()],
        y=[50, 50],
        mode='lines',
        name='Risk Threshold (50%)',
        line=dict(color=COLORS['warning'], width=3, dash="dash"),
        hoverinfo='skip'
    ))
    
    fig_risk.update_layout(
        height=400,
        margin=dict(l=0, r=20, t=30, b=60),
        xaxis_title=None,
        yaxis_title=dict(text="Risk Score (%)", font=dict(size=13, color='#1e293b')),
        yaxis=dict(range=[0, 100], tickfont=dict(size=11, color='#334155')),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, font=dict(size=12, color='#1e293b')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # Show x-axis labels every 6 months (Jan, Jul) but plot all monthly data
    fig_risk.update_xaxes(
        showgrid=False, 
        dtick='M6',  # Label every 6 months
        tickformat='%b %Y',
        range=['2022-12-15', '2026-01-15'],
        tickangle=-30,
        tickfont=dict(size=11, color='#334155')
    )
    fig_risk.update_yaxes(showgrid=True, gridcolor='#e2e8f0')
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    # --- KPI Charts ---
    st.markdown("<h4 class='section-heading'>📈 KPI Trends (Jan 2023 - Dec 2025)</h4>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    # SOTD Chart
    with col_left:
        fig_sotd = go.Figure()
        
        fig_sotd.add_trace(go.Scatter(
            x=supplier_history['date'],
            y=supplier_history['sotd_pct'],
            mode='lines',
            name='Monthly SOTD',
            line=dict(color='#60a5fa', width=2),
            hovertemplate='<b>%{x|%B %Y}</b><br>Monthly SOTD: %{y:.1f}%<extra></extra>'
        ))
        
        fig_sotd.add_trace(go.Scatter(
            x=supplier_history['date'],
            y=supplier_history['rolling_sotd_3m'],
            mode='lines+markers',
            name='3M Rolling Avg',
            line=dict(color='#1d4ed8', width=3),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%B %Y}</b><br>3M Rolling Avg: %{y:.1f}%<extra></extra>'
        ))
        
        # Threshold line with legend
        fig_sotd.add_trace(go.Scatter(
            x=[supplier_history['date'].min(), supplier_history['date'].max()],
            y=[90, 90],
            mode='lines',
            name='Risk Threshold (90%)',
            line=dict(color='#dc2626', width=2, dash="dash"),
            hoverinfo='skip'
        ))
        
        fig_sotd.update_layout(
            title=dict(text='<b>On-Time Delivery (SOTD %)</b>', font=dict(size=16, color='#1e3a5f')),
            height=380,
            margin=dict(l=0, r=0, t=50, b=70),
            xaxis_title=None,
            yaxis_title=dict(text="SOTD %", font=dict(size=12, color='#1e293b')),
            yaxis=dict(range=[75, 102], tickfont=dict(size=11, color='#334155')),
            legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5, font=dict(size=10, color='#1e293b')),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Show x-axis labels every 6 months but plot all monthly data
        fig_sotd.update_xaxes(
            showgrid=False, 
            dtick='M6',  # Label every 6 months
            tickformat='%b %Y',
            range=['2022-12-15', '2026-01-15'],
            tickangle=-30,
            tickfont=dict(size=10, color='#334155')
        )
        fig_sotd.update_yaxes(showgrid=True, gridcolor='#e2e8f0')
        
        st.plotly_chart(fig_sotd, use_container_width=True)
    
    # DPPM Chart
    with col_right:
        fig_dppm = go.Figure()
        
        fig_dppm.add_trace(go.Scatter(
            x=supplier_history['date'],
            y=supplier_history['dppm'],
            mode='lines',
            name='Monthly DPPM',
            line=dict(color='#4ade80', width=2),
            hovertemplate='<b>%{x|%B %Y}</b><br>Monthly DPPM: %{y:.0f}<extra></extra>'
        ))
        
        fig_dppm.add_trace(go.Scatter(
            x=supplier_history['date'],
            y=supplier_history['rolling_dppm_3m'],
            mode='lines+markers',
            name='3M Rolling Avg',
            line=dict(color='#15803d', width=3),
            marker=dict(size=5),
            hovertemplate='<b>%{x|%B %Y}</b><br>3M Rolling Avg: %{y:.0f}<extra></extra>'
        ))
        
        # Threshold line with legend
        fig_dppm.add_trace(go.Scatter(
            x=[supplier_history['date'].min(), supplier_history['date'].max()],
            y=[500, 500],
            mode='lines',
            name='Risk Threshold (500)',
            line=dict(color='#dc2626', width=2, dash="dash"),
            hoverinfo='skip'
        ))
        
        fig_dppm.update_layout(
            title=dict(text='<b>Defects Per Million (DPPM)</b>', font=dict(size=16, color='#1e3a5f')),
            height=380,
            margin=dict(l=0, r=0, t=50, b=70),
            xaxis_title=None,
            yaxis_title=dict(text="DPPM", font=dict(size=12, color='#1e293b')),
            yaxis=dict(tickfont=dict(size=11, color='#334155')),
            legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5, font=dict(size=10, color='#1e293b')),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Show x-axis labels every 6 months but plot all monthly data
        fig_dppm.update_xaxes(
            showgrid=False, 
            dtick='M6',  # Label every 6 months
            tickformat='%b %Y',
            range=['2022-12-15', '2026-01-15'],
            tickangle=-30,
            tickfont=dict(size=10, color='#334155')
        )
        fig_dppm.update_yaxes(showgrid=True, gridcolor='#e2e8f0')
        
        st.plotly_chart(fig_dppm, use_container_width=True)
    
    st.markdown("---")
    
    # --- Current KPIs ---
    st.markdown("<h4 class='section-heading'>📋 Current Period Summary</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sotd = supplier_latest['rolling_sotd_3m']
        st.metric(
            "SOTD (3M Avg)",
            f"{sotd:.1f}%",
            delta="⚠️" if sotd < 90 else "✓",
            delta_color="inverse" if sotd < 90 else "normal"
        )
    
    with col2:
        dppm = supplier_latest['rolling_dppm_3m']
        st.metric(
            "DPPM (3M Avg)",
            f"{dppm:.0f}",
            delta="⚠️" if dppm > 500 else "✓",
            delta_color="inverse" if dppm > 500 else "normal"
        )
    
    with col3:
        late = supplier_latest['late_delivery_rate'] * 100
        st.metric("Late Delivery Rate", f"{late:.1f}%")
    
    with col4:
        risk = supplier_latest['risk_score'] * 100
        st.metric(
            "Risk Score",
            f"{risk:.0f}%",
            delta="HIGH" if risk >= 50 else "LOW",
            delta_color="inverse" if risk >= 50 else "normal"
        )
    
    # Model info expander
    st.markdown("---")
    with st.expander("ℹ️ How Risk is Calculated"):
        st.markdown("""
        **Risk Classification Rules:**
        - High Risk: SOTD < 90% OR DPPM > 500
        - Low Risk: Otherwise
        
        **Model Features:**
        - `rolling_sotd_3m`: Lower delivery rate → Higher risk
        - `rolling_dppm_3m`: More defects → Higher risk  
        - `late_delivery_rate`: More delays → Higher risk
        
        **Model Performance:** 97.2% Accuracy
        """)


# --- Footer ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()




