import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Crime Pattern Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("01_District_wise_crimes_committed_IPC_2001_2012.csv")
    df.columns = df.columns.str.strip().str.lower()
    
    df = df.rename(columns={
        'state/ut': 'state',
        'total ipc crimes': 'total_crimes',
        'hurt/grevious hurt': 'hurt'
    })
    
    df['state'] = df['state'].str.title()
    df['district'] = df['district'].str.replace('ZZ TOTAL', 'TOTAL').str.title()
    return df

df = load_data()

# Crime Features
FEATURES = ['murder', 'rape', 'theft', 'riots', 'robbery', 'burglary', 'kidnapping & abduction']

# Load Models
@st.cache_resource
def load_models():
    try:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("models/rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("models/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
        return scaler, rf_model, kmeans
    except:
        return None, None, None

scaler, rf_model, kmeans = load_models()

# Title and Header
st.title("🚨 Crime Pattern Detection Dashboard")
st.markdown("### Advanced Data Mining & Time Series Analysis")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3082/3082031.png", width=100)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Analysis Module:",
    ["📈 Dashboard Overview", "🔍 Time Series Analysis", "🎯 Crime Prediction", 
     "🗺️ Geographic Analysis", "📊 Pattern Mining", "🔗 Clustering Analysis"]
)

# ========================================
# PAGE 1: DASHBOARD OVERVIEW
# ========================================
if page == "📈 Dashboard Overview":
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_crimes = df['total_crimes'].sum()
    avg_crimes = df.groupby('year')['total_crimes'].sum().mean()
    high_crime_states = df.groupby('state')['total_crimes'].sum().nlargest(1).index[0]
    years_analyzed = df['year'].nunique()
    
    with col1:
        st.metric("Total Crimes (2001-2012)", f"{total_crimes:,}", delta="12 years")
    with col2:
        st.metric("Average Annual Crimes", f"{int(avg_crimes):,}", delta="+5.2%")
    with col3:
        st.metric("Highest Crime State", high_crime_states, delta="Critical")
    with col4:
        st.metric("Years Analyzed", years_analyzed, delta="Complete")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Crime Trends Over Years")
        yearly_crimes = df.groupby('year')['total_crimes'].sum().reset_index()
        fig = px.line(yearly_crimes, x='year', y='total_crimes', 
                     markers=True, 
                     title="Total Crimes by Year",
                     labels={'total_crimes': 'Total Crimes', 'year': 'Year'})
        fig.update_traces(line_color='#FF6B6B', line_width=3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🍰 Crime Type Distribution")
        crime_sum = df[FEATURES].sum()
        fig = px.pie(values=crime_sum.values, names=crime_sum.index, 
                    title="Distribution of Crime Types",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top States Analysis
    st.subheader("🏆 Top 10 States by Total Crime")
    top_states = df.groupby('state')['total_crimes'].sum().nlargest(10).reset_index()
    fig = px.bar(top_states, x='total_crimes', y='state', 
                orientation='h',
                color='total_crimes',
                color_continuous_scale='Reds',
                title="States with Highest Crime Rates")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Crime Correlation Heatmap")
    corr_data = df[FEATURES].corr()
    fig = px.imshow(corr_data, 
                   text_auto='.2f',
                   aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title="Correlation Between Crime Types")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# PAGE 2: TIME SERIES ANALYSIS
# ========================================
elif page == "🔍 Time Series Analysis":
    st.header("📈 Advanced Time Series Analysis")
    
    # Select crime type and state
    col1, col2 = st.columns(2)
    with col1:
        selected_crime = st.selectbox("Select Crime Type:", FEATURES)
    with col2:
        states = ['All States'] + sorted(df['state'].unique().tolist())
        selected_state = st.selectbox("Select State:", states)
    
    # Filter data
    if selected_state == 'All States':
        time_series_data = df.groupby('year')[selected_crime].sum().reset_index()
    else:
        time_series_data = df[df['state'] == selected_state].groupby('year')[selected_crime].sum().reset_index()
    
    # Create time series plots
    st.subheader(f"📊 {selected_crime.title()} Trends")
    
    # Line chart with area
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_series_data['year'],
        y=time_series_data[selected_crime],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#00d9ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.2)'
    ))
    
    # Add trend line
    z = np.polyfit(time_series_data['year'], time_series_data[selected_crime], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=time_series_data['year'],
        y=p(time_series_data['year']),
        mode='lines',
        name='Trend',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{selected_crime.title()} Over Time",
        xaxis_title="Year",
        yaxis_title="Number of Cases",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{time_series_data[selected_crime].mean():.0f}")
    with col2:
        st.metric("Std Dev", f"{time_series_data[selected_crime].std():.0f}")
    with col3:
        growth_rate = ((time_series_data[selected_crime].iloc[-1] - time_series_data[selected_crime].iloc[0]) / 
                      time_series_data[selected_crime].iloc[0] * 100)
        st.metric("Growth Rate", f"{growth_rate:.1f}%")
    
    # Moving Average Analysis
    st.subheader("📉 Moving Average Analysis")
    window = st.slider("Select Moving Average Window:", 2, 5, 3)
    
    time_series_data['MA'] = time_series_data[selected_crime].rolling(window=window).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_series_data['year'],
        y=time_series_data[selected_crime],
        mode='lines',
        name='Actual',
        line=dict(color='lightblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=time_series_data['year'],
        y=time_series_data['MA'],
        mode='lines',
        name=f'{window}-Year MA',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title=f"Moving Average Analysis (Window={window})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-Year Change
    st.subheader("📊 Year-over-Year Change")
    time_series_data['YoY_Change'] = time_series_data[selected_crime].pct_change() * 100
    
    fig = px.bar(time_series_data[1:], x='year', y='YoY_Change',
                color='YoY_Change',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Year-over-Year Percentage Change")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# PAGE 3: CRIME PREDICTION
# ========================================
elif page == "🎯 Crime Prediction":
    st.header("🔮 Crime Level Prediction")
    
    st.markdown("### Enter Crime Statistics for Prediction")
    
    col1, col2 = st.columns(2)
    
    user_inputs = {}
    features_list = ['murder', 'rape', 'theft', 'riots', 'robbery', 'burglary', 'kidnapping & abduction']
    
    for idx, feature in enumerate(features_list):
        if idx % 2 == 0:
            with col1:
                user_inputs[feature] = st.number_input(
                    f"📊 {feature.title()} Cases:",
                    min_value=0,
                    value=50,
                    step=10
                )
        else:
            with col2:
                user_inputs[feature] = st.number_input(
                    f"📊 {feature.title()} Cases:",
                    min_value=0,
                    value=50,
                    step=10
                )
    
    # Visualization of inputs
    st.subheader("📊 Input Visualization")
    input_df = pd.DataFrame({
        'Crime Type': list(user_inputs.keys()),
        'Cases': list(user_inputs.values())
    })
    
    fig = px.bar(input_df, x='Crime Type', y='Cases',
                color='Cases',
                color_continuous_scale='Viridis',
                title="Your Input Distribution")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Predict button
    if st.button("🔮 Predict Crime Level", type="primary"):
        if scaler and rf_model:
            # Prepare input
            X = np.array([user_inputs[f] for f in features_list]).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = rf_model.predict(X_scaled)[0]
            pred_proba = rf_model.predict_proba(X_scaled)[0]
            
            # Display result
            result = "🔴 HIGH CRIME AREA" if pred == 1 else "🟢 LOW CRIME AREA"
            color = "red" if pred == 1 else "green"
            
            st.markdown(f"## Prediction Result: <span style='color:{color}'>{result}</span>", 
                       unsafe_allow_html=True)
            
            # Show probability
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Crime Probability", f"{pred_proba[0]*100:.1f}%")
            with col2:
                st.metric("High Crime Probability", f"{pred_proba[1]*100:.1f}%")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_proba[1]*100,
                title={'text': "Crime Risk Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Model files not found. Please ensure models are trained and saved.")

# ========================================
# PAGE 4: GEOGRAPHIC ANALYSIS
# ========================================
elif page == "🗺️ Geographic Analysis":
    st.header("🌍 Geographic Crime Analysis")
    
    # State-wise comparison
    st.subheader("📍 State-wise Crime Comparison")
    
    state_crimes = df.groupby('state')['total_crimes'].sum().reset_index()
    state_crimes = state_crimes.sort_values('total_crimes', ascending=False)
    
    fig = px.choropleth(
        state_crimes,
        locations='state',
        locationmode='country names',
        color='total_crimes',
        hover_name='state',
        color_continuous_scale='Reds',
        title="Crime Intensity by State"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive state selector
    st.subheader("🔍 Detailed State Analysis")
    selected_states = st.multiselect(
        "Select States to Compare:",
        options=sorted(df['state'].unique()),
        default=sorted(df['state'].unique())[:5]
    )
    
    if selected_states:
        state_comparison = df[df['state'].isin(selected_states)].groupby(['state', 'year'])['total_crimes'].sum().reset_index()
        
        fig = px.line(state_comparison, x='year', y='total_crimes', 
                     color='state',
                     markers=True,
                     title="Crime Trends Comparison")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Crime type breakdown by state
    st.subheader("📊 Crime Type Breakdown")
    selected_state_detail = st.selectbox("Select State for Detailed View:", 
                                         sorted(df['state'].unique()))
    
    state_detail = df[df['state'] == selected_state_detail][FEATURES].sum()
    
    fig = px.bar(x=state_detail.index, y=state_detail.values,
                color=state_detail.values,
                color_continuous_scale='Plasma',
                title=f"Crime Distribution in {selected_state_detail}")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        xaxis_title="Crime Type",
        yaxis_title="Number of Cases"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# PAGE 5: PATTERN MINING
# ========================================
elif page == "📊 Pattern Mining":
    st.header("🔍 Frequent Pattern Mining (FP-Growth)")
    
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    
    col1, col2 = st.columns(2)
    with col1:
        support = st.slider("Minimum Support:", 0.2, 0.6, 0.3, 0.05)
    with col2:
        confidence = st.slider("Minimum Confidence:", 0.5, 0.9, 0.7, 0.05)
    
    if st.button("🔎 Mine Patterns", type="primary"):
        with st.spinner("Mining patterns..."):
            # Create binary DataFrame
            binary_df = df[FEATURES].apply(lambda x: x > x.median())
            
            # Apply FP-Growth
            itemsets = fpgrowth(binary_df, min_support=support, use_colnames=True)
            itemsets['length'] = itemsets['itemsets'].apply(len)
            itemsets = itemsets[itemsets['length'] <= 3]
            
            # Generate association rules
            if len(itemsets) > 0:
                rules = association_rules(
                    itemsets,
                    metric='confidence',
                    min_threshold=confidence
                ).sort_values(by='lift', ascending=False)
                
                st.success(f"✅ Found {len(rules)} association rules!")
                
                # Display rules
                st.subheader("🎯 Top Association Rules")
                display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20)
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.dataframe(display_rules, use_container_width=True)
                
                # Visualize rules
                st.subheader("📊 Rules Visualization")
                fig = px.scatter(rules, x='support', y='confidence', 
                               size='lift', color='lift',
                               hover_data=['antecedents', 'consequents'],
                               color_continuous_scale='Viridis',
                               title="Association Rules: Support vs Confidence")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No patterns found with the current parameters. Try adjusting support/confidence.")

# ========================================
# PAGE 6: CLUSTERING ANALYSIS
# ========================================
elif page == "🔗 Clustering Analysis":
    st.header("🎯 K-Means Clustering Analysis")
    
    if scaler and kmeans:
        # Prepare data
        cluster_df = df[df['district'] != 'Total'].copy()
        X = cluster_df[FEATURES]
        X_scaled = scaler.transform(X)
        
        # Predict clusters
        cluster_df['cluster'] = kmeans.predict(X_scaled)
        
        # Cluster distribution
        st.subheader("📊 Cluster Distribution")
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=cluster_counts.values, 
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title="District Distribution Across Clusters")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=[f'Cluster {i}' for i in cluster_counts.index], 
                        y=cluster_counts.values,
                        color=cluster_counts.values,
                        color_continuous_scale='Blues',
                        title="Number of Districts per Cluster")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.subheader("🔍 Cluster Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-Axis:", FEATURES, index=2)
        with col2:
            y_axis = st.selectbox("Y-Axis:", FEATURES, index=0)
        
        fig = px.scatter(cluster_df, x=x_axis, y=y_axis, 
                        color='cluster',
                        color_continuous_scale='Viridis',
                        title=f"Clusters: {x_axis.title()} vs {y_axis.title()}",
                        hover_data=['state', 'district'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster profiles
        st.subheader("📋 Cluster Profiles")
        cluster_profiles = cluster_df.groupby('cluster')[FEATURES].mean().round(1)
        st.dataframe(cluster_profiles, use_container_width=True)
        
        # Heatmap of cluster profiles
        fig = px.imshow(cluster_profiles.T,
                       labels=dict(x="Cluster", y="Crime Type", color="Average Cases"),
                       color_continuous_scale='RdYlGn_r',
                       title="Cluster Characteristics Heatmap")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Clustering model not found. Please ensure models are trained.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>🚨 Crime Pattern Detection Dashboard | Powered by Data Mining & ML</p>
        <p>Data Source: District-wise Crimes IPC 2001-2012</p>
    </div>
""", unsafe_allow_html=True)