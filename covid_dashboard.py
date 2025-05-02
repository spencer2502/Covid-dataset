import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Multi-Country Time Series Visualizer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Step 1: Load and Cache Data -------------------
@st.cache_data
def load_data():
    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recovered_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    
    # Get the latest daily report
    current_date = datetime.datetime.now() - datetime.timedelta(days=2)  # Use 2 days ago to ensure data is available
    date_str = current_date.strftime('%m-%d-%Y')
    try:
        latest_data = pd.read_csv(
            f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv')
    except:
        # If current date isn't available, try a day earlier
        current_date = current_date - datetime.timedelta(days=1)
        date_str = current_date.strftime('%m-%d-%Y')
        try:
            latest_data = pd.read_csv(
                f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv')
        except:
            # If still not available, use a default date
            latest_data = pd.read_csv(
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-15-2023.csv')
    
    return confirmed_df, deaths_df, recovered_df, latest_data

# Load data
with st.spinner('Loading data...'):
    confirmed_df, deaths_df, recovered_df, latest_data = load_data()

# ------------------- Step 2: Sidebar for User Controls -------------------
st.sidebar.title("COVID-19 Dashboard Controls")

# Country selection
countries = list(confirmed_df['Country/Region'].unique())
default_countries = ['US', 'India', 'Brazil', 'United Kingdom', 'France']
selected_countries = st.sidebar.multiselect('Select Countries to Visualize', countries, default=default_countries)

# Display mode selection
display_mode = st.sidebar.radio(
    "Select Display Mode",
    ('Global Overview', 'Country Comparison', 'Prediction Models')
)

# Parameters for analysis
window_size = st.sidebar.slider("Rolling Average Window Size", 3, 21, 7)
days_to_forecast = st.sidebar.slider("Days to Forecast", 7, 60, 30)

# ------------------- Step 3: Data Preprocessing Functions -------------------
def preprocess_data():
    # Extract date columns
    confirmed = confirmed_df.iloc[:, 4:]
    deaths = deaths_df.iloc[:, 4:]
    recovered = recovered_df.iloc[:, 4:]
    
    # Get dates
    dates = confirmed.columns
    date_list = [datetime.datetime.strptime(date, '%m/%d/%y').strftime('%Y-%m-%d') for date in dates]
    
    # Calculate global cases
    world_cases = confirmed.sum(axis=0).values
    total_deaths = deaths.sum(axis=0).values
    total_recovered = recovered.sum(axis=0).values
    
    # Calculate derived metrics
    active_cases = world_cases - total_deaths - total_recovered
    mortality_rate = (total_deaths / world_cases) * 100
    recovery_rate = (total_recovered / world_cases) * 100
    
    # Create global dataframe
    global_df = pd.DataFrame({
        'Date': date_list,
        'Confirmed': world_cases,
        'Deaths': total_deaths,
        'Recovered': total_recovered,
        'Active': active_cases,
        'Mortality_Rate': mortality_rate,
        'Recovery_Rate': recovery_rate
    })
    
    global_df['Date'] = pd.to_datetime(global_df['Date'])
    
    return global_df, date_list

def get_country_data(country_name):
    # Filter data for the selected country
    country_confirmed = confirmed_df[confirmed_df['Country/Region'] == country_name].iloc[:, 4:].sum(axis=0).values
    country_deaths = deaths_df[deaths_df['Country/Region'] == country_name].iloc[:, 4:].sum(axis=0).values
    country_recovered = recovered_df[recovered_df['Country/Region'] == country_name].iloc[:, 4:].sum(axis=0).values
    
    # Get dates
    dates = confirmed_df.iloc[:, 4:].columns
    date_list = [datetime.datetime.strptime(date, '%m/%d/%y').strftime('%Y-%m-%d') for date in dates]
    
    # Calculate derived metrics
    active_cases = country_confirmed - country_deaths - country_recovered
    mortality_rate = (country_deaths / country_confirmed) * 100
    recovery_rate = (country_recovered / country_confirmed) * 100
    
    # Create country dataframe
    country_df = pd.DataFrame({
        'Date': date_list,
        'Confirmed': country_confirmed,
        'Deaths': country_deaths,
        'Recovered': country_recovered,
        'Active': active_cases,
        'Mortality_Rate': mortality_rate,
        'Recovery_Rate': recovery_rate
    })
    
    country_df['Date'] = pd.to_datetime(country_df['Date'])
    
    return country_df

def daily_increase(data):
    """Calculate daily increase from cumulative data"""
    return [data[i] - data[i-1] if i > 0 else data[0] for i in range(len(data))]

def moving_average(data, window=7):
    """Calculate moving average with specified window size"""
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

# ------------------- Step 4: Process Global Data -------------------
global_df, date_list = preprocess_data()

# Add daily increase and moving averages to global data
global_df['Daily_Confirmed'] = daily_increase(global_df['Confirmed'].values)
global_df['Daily_Deaths'] = daily_increase(global_df['Deaths'].values)
global_df['Daily_Recovered'] = daily_increase(global_df['Recovered'].values)

global_df['Confirmed_MA'] = moving_average(global_df['Confirmed'].values, window_size)
global_df['Deaths_MA'] = moving_average(global_df['Deaths'].values, window_size)
global_df['Recovered_MA'] = moving_average(global_df['Recovered'].values, window_size)
global_df['Daily_Confirmed_MA'] = moving_average(global_df['Daily_Confirmed'].values, window_size)
global_df['Daily_Deaths_MA'] = moving_average(global_df['Daily_Deaths'].values, window_size)
global_df['Daily_Recovered_MA'] = moving_average(global_df['Daily_Recovered'].values, window_size)

# ------------------- Step 5: Main Dashboard -------------------
st.title("🦠 COVID-19 Multi-Country Time Series Visualizer")
st.markdown("""
This dashboard visualizes COVID-19 data across multiple countries, including confirmed cases, 
deaths, recoveries, and predictive models. Use the sidebar to select countries and adjust parameters.
""")

# ------------------- Step 6: Global Overview -------------------
if display_mode == 'Global Overview':
    st.header("Global COVID-19 Overview")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Confirmed Cases", f"{global_df['Confirmed'].iloc[-1]:,}", 
                 f"{global_df['Daily_Confirmed'].iloc[-1]:,}")
    
    with col2:
        st.metric("Total Deaths", f"{global_df['Deaths'].iloc[-1]:,}", 
                 f"{global_df['Daily_Deaths'].iloc[-1]:,}")
    
    with col3:
        st.metric("Total Recovered", f"{global_df['Recovered'].iloc[-1]:,}", 
                 f"{global_df['Daily_Recovered'].iloc[-1]:,}")
    
    with col4:
        st.metric("Current Active Cases", f"{global_df['Active'].iloc[-1]:,}", 
                 f"{global_df['Active'].iloc[-1] - global_df['Active'].iloc[-2]:,}")
    
    # Global trend charts
    st.subheader("Global COVID-19 Trends")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Cases", "Daily Cases", "Mortality & Recovery", "Geographic Distribution"])
    
    with tab1:
        # Cumulative cases plot
        fig = px.line(global_df, x='Date', y=['Confirmed', 'Deaths', 'Recovered'],
                     title='Global Cumulative COVID-19 Cases',
                     labels={'value': 'Number of Cases', 'variable': 'Category'},
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Daily cases with moving average
        fig = px.line(global_df, x='Date', y=['Daily_Confirmed_MA', 'Daily_Deaths_MA', 'Daily_Recovered_MA'],
                     title=f'Global Daily COVID-19 Cases ({window_size}-Day Moving Average)',
                     labels={'value': 'Number of Cases', 'variable': 'Category'},
                     color_discrete_sequence=px.colors.qualitative.Bold)
        
        # Find and annotate peak days
        peak_cases_idx = global_df['Daily_Confirmed_MA'].idxmax()
        peak_deaths_idx = global_df['Daily_Deaths_MA'].idxmax()
        
        fig.add_annotation(
            x=global_df.iloc[peak_cases_idx]['Date'],
            y=global_df.iloc[peak_cases_idx]['Daily_Confirmed_MA'],
            text=f"Peak Cases: {int(global_df.iloc[peak_cases_idx]['Daily_Confirmed_MA']):,}",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=global_df.iloc[peak_deaths_idx]['Date'],
            y=global_df.iloc[peak_deaths_idx]['Daily_Deaths_MA'],
            text=f"Peak Deaths: {int(global_df.iloc[peak_deaths_idx]['Daily_Deaths_MA']):,}",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Mortality and recovery rates
        fig = px.line(global_df, x='Date', y=['Mortality_Rate', 'Recovery_Rate'],
                     title='Global COVID-19 Mortality and Recovery Rates',
                     labels={'value': 'Rate (%)', 'variable': 'Category'},
                     color_discrete_sequence=['red', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Geographic distribution from latest data
        st.write("COVID-19 Geographic Distribution (Latest Data)")
        
        # Prepare data for map
        map_data = latest_data.groupby('Country_Region').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum',
            'Recovered': 'sum',
            'Active': 'sum',
            'Lat': 'mean',
            'Long_': 'mean'
        }).reset_index()
        
        # Create map
        fig = px.scatter_geo(map_data, 
                            lat='Lat', 
                            lon='Long_', 
                            size='Confirmed',
                            color='Confirmed',
                            hover_name='Country_Region',
                            hover_data=['Confirmed', 'Deaths', 'Recovered', 'Active'],
                            projection='natural earth',
                            title='Global COVID-19 Case Distribution',
                            color_continuous_scale=px.colors.sequential.Plasma)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Dual Y-Axis Plot
    st.subheader("Dual Y-Axis Plot: Active vs Total Cases")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=global_df['Date'], y=global_df['Confirmed'], name="Total Cases"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=global_df['Date'], y=global_df['Active'], name="Active Cases"),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Global COVID-19: Total vs Active Cases",
        xaxis_title="Date",
    )
    
    fig.update_yaxes(title_text="Total Cases", secondary_y=False)
    fig.update_yaxes(title_text="Active Cases", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Step 7: Country Comparison -------------------
elif display_mode == 'Country Comparison':
    st.header("Country Comparison")
    
    if not selected_countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Prepare data for all selected countries
        country_data = {}
        for country in selected_countries:
            country_data[country] = get_country_data(country)
        
        # Create comparative visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Confirmed Cases", "Deaths", "Active Cases", "Comparative Metrics"])
        
        with tab1:
            st.subheader("Confirmed Cases Comparison")
            
            # Line chart for confirmed cases
            fig = go.Figure()
            for country in selected_countries:
                df = country_data[country]
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Confirmed'], mode='lines', name=country))
            
            fig.update_layout(title="Confirmed COVID-19 Cases by Country",
                             xaxis_title="Date",
                             yaxis_title="Confirmed Cases",
                             height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily new cases with moving average
            st.subheader(f"Daily New Cases ({window_size}-Day Moving Average)")
            fig = go.Figure()
            for country in selected_countries:
                df = country_data[country]
                daily_cases = daily_increase(df['Confirmed'].values)
                daily_ma = moving_average(daily_cases, window_size)
                
                temp_df = pd.DataFrame({
                    'Date': df['Date'],
                    'Daily_MA': daily_ma
                })
                
                fig.add_trace(go.Scatter(x=temp_df['Date'], y=temp_df['Daily_MA'], mode='lines', name=country))
            
            fig.update_layout(title=f"Daily New COVID-19 Cases ({window_size}-Day Moving Average)",
                             xaxis_title="Date",
                             yaxis_title="Daily New Cases",
                             height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Deaths Comparison")
            
            # Line chart for deaths
            fig = go.Figure()
            for country in selected_countries:
                df = country_data[country]
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Deaths'], mode='lines', name=country))
            
            fig.update_layout(title="COVID-19 Deaths by Country",
                             xaxis_title="Date",
                             yaxis_title="Deaths",
                             height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mortality rate comparison
            st.subheader("Mortality Rate Comparison")
            fig = go.Figure()
            for country in selected_countries:
                df = country_data[country]
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Mortality_Rate'], mode='lines', name=country))
            
            fig.update_layout(title="COVID-19 Mortality Rate by Country",
                             xaxis_title="Date",
                             yaxis_title="Mortality Rate (%)",
                             height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Active Cases Comparison")
            
            # Line chart for active cases
            fig = go.Figure()
            for country in selected_countries:
                df = country_data[country]
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Active'], mode='lines', name=country))
            
            fig.update_layout(title="Active COVID-19 Cases by Country",
                             xaxis_title="Date",
                             yaxis_title="Active Cases",
                             height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Comparative Metrics")
            
            # Create a dataframe for comparison
            latest_data = {}
            for country in selected_countries:
                df = country_data[country]
                latest = df.iloc[-1]
                latest_data[country] = {
                    'Confirmed': latest['Confirmed'],
                    'Deaths': latest['Deaths'],
                    'Recovered': latest['Recovered'],
                    'Active': latest['Active'],
                    'Mortality_Rate': latest['Mortality_Rate'],
                    'Recovery_Rate': latest['Recovery_Rate']
                }
            
            comparison_df = pd.DataFrame(latest_data).T
            comparison_df = comparison_df.reset_index().rename(columns={'index': 'Country'})
            
            # Bar charts for comparison
            metrics = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            
            for metric in metrics:
                fig = px.bar(comparison_df, x='Country', y=metric, 
                             title=f"{metric} Cases by Country",
                             color='Country')
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart for combined metrics
            fig = go.Figure()
            
            for country in selected_countries:
                fig.add_trace(go.Scatterpolar(
                    r=[
                        comparison_df[comparison_df['Country'] == country]['Confirmed'].values[0] / 1e6,
                        comparison_df[comparison_df['Country'] == country]['Deaths'].values[0] / 1e4,
                        comparison_df[comparison_df['Country'] == country]['Recovered'].values[0] / 1e6,
                        comparison_df[comparison_df['Country'] == country]['Active'].values[0] / 1e5,
                        comparison_df[comparison_df['Country'] == country]['Mortality_Rate'].values[0],
                    ],
                    theta=['Confirmed (millions)', 'Deaths (10,000s)', 'Recovered (millions)', 
                           'Active (100,000s)', 'Mortality Rate (%)'],
                    fill='toself',
                    name=country
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )),
                showlegend=True,
                title="Multi-dimensional COVID-19 Metric Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ------------------- Step 8: Prediction Models -------------------
else:  # Prediction Models
    st.header("COVID-19 Prediction Models")
    
    if not selected_countries:
        st.warning("Please select at least one country from the sidebar.")
        country_for_prediction = "US"  # Default
    else:
        country_for_prediction = st.selectbox("Select a country for prediction", selected_countries)
    
    # Get data for the selected country
    country_df = get_country_data(country_for_prediction)
    
    # Prepare data for modeling
    X = np.array(range(len(country_df))).reshape(-1, 1)
    y = np.array(country_df['Confirmed'].values).reshape(-1, 1)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Setup future dates for prediction
    last_date = country_df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=days_to_forecast)
    future_X = np.array(range(len(country_df), len(country_df) + days_to_forecast)).reshape(-1, 1)
    
    # Combine current and future dates
    all_X = np.vstack([X, future_X])
    # Create continuous date range
    date_range = pd.to_datetime(pd.concat([
        pd.Series(country_df['Date'].values),
        pd.Series(future_dates)
    ]).reset_index(drop=True))
    
    # Choose prediction model
    model_type = st.selectbox(
        "Select Prediction Model",
        ("Polynomial Regression", "Bayesian Ridge", "Support Vector Regression")
    )
    
    # Fit the selected model
    if model_type == "Polynomial Regression":
        # Polynomial features
        poly = PolynomialFeatures(degree=3)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        X_poly_all = poly.transform(all_X)
        
        # Linear regression on polynomial features
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_poly_train)
        test_preds = model.predict(X_poly_test)
        all_preds = model.predict(X_poly_all)
        
        # Evaluate model
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
    elif model_type == "Bayesian Ridge":
        # Polynomial features for Bayesian Ridge
        poly = PolynomialFeatures(degree=3)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        X_poly_all = poly.transform(all_X)
        
        # Bayesian Ridge regression
        model = BayesianRidge()
        model.fit(X_poly_train, y_train.ravel())
        
        # Make predictions
        train_preds = model.predict(X_poly_train).reshape(-1, 1)
        test_preds = model.predict(X_poly_test).reshape(-1, 1)
        all_preds = model.predict(X_poly_all).reshape(-1, 1)
        
        # Evaluate model
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
    else:  # Support Vector Regression
        # SVR model
        model = SVR(kernel='poly', C=0.1, gamma=0.01, epsilon=1, degree=3)
        model.fit(X_train, y_train.ravel())
        
        # Make predictions
        train_preds = model.predict(X_train).reshape(-1, 1)
        test_preds = model.predict(X_test).reshape(-1, 1)
        all_preds = model.predict(all_X).reshape(-1, 1)
        
        # Evaluate model
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    # Display model performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training RMSE", f"{train_rmse:,.2f}")
    with col2:
        st.metric("Testing RMSE", f"{test_rmse:,.2f}")
    
    # Create prediction dataframe 
    # Handle actual data (with NaN padding for future dates)
    actual_data = np.vstack([y, np.array([[np.nan]] * days_to_forecast)]).flatten()
    
    pred_df = pd.DataFrame({
        'Date': date_range,
        'Actual': actual_data,
        'Predicted': all_preds.flatten()
    })
    
    # Plot predictions
    st.subheader(f"COVID-19 Case Predictions for {country_for_prediction}")
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(
        go.Scatter(
            x=pred_df['Date'][:-days_to_forecast],
            y=pred_df['Actual'][:-days_to_forecast],
            mode='lines',
            name='Actual Cases',
            line=dict(color='blue')
        )
    )
    
    # Predicted data (historical)
    fig.add_trace(
        go.Scatter(
            x=pred_df['Date'][:-days_to_forecast],
            y=pred_df['Predicted'][:-days_to_forecast],
            mode='lines',
            name='Model Fit',
            line=dict(color='green', dash='dash')
        )
    )
    
    # Future predictions
    fig.add_trace(
        go.Scatter(
            x=pred_df['Date'][-days_to_forecast:],
            y=pred_df['Predicted'][-days_to_forecast:],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Add vertical line at current date - using timestamp in milliseconds to avoid datetime issues
    fig.add_shape(
        type="line",
        x0=country_df['Date'].iloc[-1],
        x1=country_df['Date'].iloc[-1],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="black", width=2, dash="dash"),
    )
    
    # Add annotation for the forecast start
    fig.add_annotation(
        x=country_df['Date'].iloc[-1],
        y=0.95,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="black"),
        align="center"
    )
    
    fig.update_layout(
        title=f"{model_type} Forecast for {country_for_prediction}",
        xaxis_title="Date",
        yaxis_title="Confirmed Cases",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecasted values in table
    st.subheader("Forecasted Cases")
    forecast_table = pd.DataFrame({
        'Date': pred_df['Date'][-days_to_forecast:],
        'Forecasted Cases': pred_df['Predicted'][-days_to_forecast:].astype(int)
    })
    
    st.dataframe(forecast_table.set_index('Date'), use_container_width=True)

# ------------------- Step 9: Footer -------------------
st.markdown("---")
st.markdown("""
*This dashboard is for educational purposes only. Data source: Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE).*

*Last updated: {}*
""".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))