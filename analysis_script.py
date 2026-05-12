import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
import datetime
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from scipy.interpolate import lagrange

# Create figures directory
if not os.path.exists('figures'):
    os.makedirs('figures')

def load_and_prep_data():
    print("Pulling S&P 500 and VIX data using yfinance...")
    start_date = "2013-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    # Pull data using yfinance instead of FRED
    df_sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    df_vix = yf.download('^VIX', start=start_date, end=end_date)
    
    # Combine into one dataframe
    df_market = pd.DataFrame()
    df_market['SP500'] = df_sp500['Close'].squeeze()
    df_market['VIXCLS'] = df_vix['Close'].squeeze()
    df_market.dropna(inplace=True)
    
    # Calculate daily returns
    df_market['SP500_Return'] = df_market['SP500'].pct_change() * 100
    df_market.dropna(inplace=True)
    
    # Load events
    print("Loading events.csv...")
    df_events = pd.read_csv('events.csv', parse_dates=['event_date'])
    
    # Align dates (if after market close, shift to next day)
    df_events['effective_date'] = df_events.apply(
        lambda row: row['event_date'] + pd.Timedelta(days=1) if row['after_market_close'] else row['event_date'], 
        axis=1
    )
    
    # Function to find the next available trading day
    def get_next_trading_day(date, trading_days):
        while date not in trading_days:
            date += pd.Timedelta(days=1)
            if date > trading_days.max():
                return pd.NaT
        return date

    # Ensure timezones match between datasets
    df_market.index = df_market.index.tz_localize(None)
    trading_days = df_market.index
    
    df_events['trading_date'] = df_events['effective_date'].apply(lambda x: get_next_trading_day(x, trading_days))
    df_events.dropna(subset=['trading_date'], inplace=True)
    
    # Flag event days in the market dataframe
    df_market['Is_Event_Day'] = df_market.index.isin(df_events['trading_date'])
    
    return df_market, df_events

def run_eda(df_market):
    print("Running Exploratory Data Analysis...")
    
    # 1. Price Series
    plt.figure(figsize=(12, 6))
    plt.plot(df_market.index, df_market['SP500'], label='S&P 500', color='blue')
    plt.title('S&P 500 Price Series (2013 - Present)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.savefig('figures/sp500_price.png')
    plt.close()

    # 2. Daily Returns
    plt.figure(figsize=(12, 6))
    plt.plot(df_market.index, df_market['SP500_Return'], label='Daily Return (%)', color='gray', alpha=0.7)
    plt.title('S&P 500 Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.axhline(0, color='black', linestyle='--')
    plt.savefig('figures/sp500_returns.png')
    plt.close()

    # 3. Boxplot: Event vs Non-Event
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Is_Event_Day', y='SP500_Return', data=df_market)
    plt.title('S&P 500 Returns: Event Days vs Non-Event Days')
    plt.xticks([0, 1], ['Non-Event', 'Conflict Event'])
    plt.ylabel('Daily Return (%)')
    plt.savefig('figures/returns_boxplot.png')
    plt.close()

def run_hypothesis_test(df_market):
    print("\nRunning Hypothesis Test...")
    event_returns = df_market[df_market['Is_Event_Day']]['SP500_Return']
    non_event_returns = df_market[~df_market['Is_Event_Day']]['SP500_Return']
    
    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(event_returns, non_event_returns, equal_var=False)
    
    print(f"H0: Major conflicts do NOT significantly affect short-term S&P 500 returns.")
    print(f"H1: Major conflicts DO significantly affect short-term S&P 500 returns.")
    print("-" * 30)
    print(f"Event Day Mean Return: {event_returns.mean():.4f}%")
    print(f"Non-Event Day Mean Return: {non_event_returns.mean():.4f}%")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Reject H0. There is a statistically significant difference in returns.")
    else:
        print("Result: Fail to reject H0. No statistically significant difference in returns at the 5% level.")

def print_top_10_spikes(df_market, df_events):
    print("\n" + "="*50)
    print("TOP 10 MARKET SPIKES ON CONFLICT DAYS")
    print("="*50)
    
    # Extract just the returns and dates from the market data
    df_market_returns = df_market[['SP500_Return']].copy()
    df_market_returns['trading_date'] = df_market_returns.index
    
    # Merge the event text with the actual market return numbers
    merged_data = pd.merge(df_events, df_market_returns, on='trading_date', how='inner')
    
    # Sort to find the highest spikes
    top_10 = merged_data.sort_values(by='SP500_Return', ascending=False).head(10)
    
    rank = 1
    for index, row in top_10.iterrows():
        date_str = row['event_date'].strftime('%Y-%m-%d')
        headline = row['headline']
        spike_pct = row['SP500_Return']
        
        print(f"#{rank} | {date_str} | Spike: +{spike_pct:.2f}%")
        print(f"    Event: {headline}\n")
        rank += 1

def apply_lagrange_interpolation(df_market):
    print("Running Lagrange Interpolation for Weekend Crisis Estimation...")
    
    # We find indices where VIX is missing (weekends/holidays) 
    # but an event occurred.
    for i in range(2, len(df_market) - 2):
        if df_market['Is_Event_Day'].iloc[i] == True and pd.isna(df_market['VIXCLS'].iloc[i]):
            
            # Grab surrounding x (indices) and y (VIX values)
            # We'll use 2 days before and 2 days after to build the polynomial
            x_known = np.array([i-2, i-1, i+1, i+2])
            y_known = df_market['VIXCLS'].iloc[x_known].values
            
            # Create the Lagrange Polynomial
            poly = lagrange(x_known, y_known)
            
            # Estimate the VIX for the missing day (index i)
            estimated_vix = poly(i)
            
            # Fill the missing value
            df_market.at[df_market.index[i], 'VIXCLS'] = estimated_vix
            
    print("Weekend gaps bridged using Lagrange Polynomials.")
    return df_market

def apply_simpsons_rule(df_market):
    print("Applying Simpson's 1/3 Rule to calculate Total Market Stress...")
    
    # Create a new column full of zeros to hold our new math
    df_market['Total_Market_Stress'] = 0.0
    
    # We use h = 1 because our time steps are 1 day
    h = 1.0 
    
    # Loop through the data to find Event Days
    for i in range(1, len(df_market) - 1):
        if df_market['Is_Event_Day'].iloc[i] == True:
            # Get the VIX for the day before, the day of, and the day after
            vix_t_minus_1 = df_market['VIXCLS'].iloc[i-1]
            vix_t_0       = df_market['VIXCLS'].iloc[i]
            vix_t_plus_1  = df_market['VIXCLS'].iloc[i+1]
            
            # Apply Simpson's 1/3 Rule Formula
            area_under_curve = (h / 3) * (vix_t_minus_1 + 4 * vix_t_0 + vix_t_plus_1)
            
            # Store the result
            df_market.at[df_market.index[i], 'Total_Market_Stress'] = area_under_curve
            
    print("Numerical Integration complete. New feature 'Total_Market_Stress' added.")
    return df_market

def run_ml_clustering(df_market):
    print("\nRunning ML K-Means Clustering...")
    
    # Prepare data
    ml_data = df_market[['SP500_Return', 'VIXCLS']].dropna().copy()
    
    # Scale data 
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(ml_data)
    
    # Run the K-Means algorithm to find 3 market regimes
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    ml_data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Add our event flag back in
    ml_data['Is_Event_Day'] = df_market.loc[ml_data.index, 'Is_Event_Day']
    
    # --- NEW PLOTTING LOGIC ---
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the "Normal" days first, slightly faded so the events stand out
    normal_days = ml_data[~ml_data['Is_Event_Day']]
    sns.scatterplot(
        x='VIXCLS', y='SP500_Return', 
        hue='Cluster', palette='viridis', data=normal_days, 
        s=60, alpha=0.3, legend='full'
    )
    
    # 2. Plot the "Event" days on top in BRIGHT RED
    event_days = ml_data[ml_data['Is_Event_Day']]
    plt.scatter(
        event_days['VIXCLS'], event_days['SP500_Return'], 
        color='red', marker='X', s=120, edgecolor='black', 
        label='Conflict Event', zorder=5
    )
    
    plt.title('ML Market Regimes: S&P 500 Returns vs VIX (Fear Gauge)')
    plt.xlabel('VIX (Market Volatility)')
    plt.ylabel('S&P 500 Daily Return (%)')
    
    # Fix the legend so it includes the red X
    plt.legend(loc='lower right')
    
    plt.savefig('figures/ml_clusters.png')
    plt.close()
    print("ML Clustering complete. Saved to 'figures/ml_clusters.png'")

def run_supervised_ml(df_market):
    print("\nRunning Supervised ML with SMOTE...")
    
    # 1. Prepare the Data
    ml_data = df_market[['SP500_Return', 'VIXCLS', 'Is_Event_Day']].dropna()
    X = ml_data[['SP500_Return', 'VIXCLS']]
    y = ml_data['Is_Event_Day'].astype(int)
    
    # 2. Split into Training Data and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Apply SMOTE ONLY to the training data
    print("Synthesizing new Event Days using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 4. Train the Model on the new, balanced data
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_smote, y_train_smote)
    
    # 5. Make Predictions on the UNSEEN Test Data
    y_pred = rf_model.predict(X_test)
    
    # 6. Evaluate and Visualize
    print("\nModel Evaluation (After SMOTE):")
    print(classification_report(y_test, y_pred, target_names=['Normal Day', 'Event Day']))
    
    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        rf_model, X_test, y_test,
        display_labels=['Normal Day', 'Event Day'], 
        cmap='Blues', colorbar=False
    )
    plt.title("Supervised ML: Predictions (With SMOTE)")
    plt.savefig('figures/supervised_confusion_matrix.png')
    plt.close()
    
    print("Supervised ML complete. Saved to 'figures/supervised_confusion_matrix.png'")

if __name__ == "__main__":
    market_data, event_data = load_and_prep_data()
    run_eda(market_data)
    run_hypothesis_test(market_data)
    print_top_10_spikes(market_data, event_data)
    market_data = apply_lagrange_interpolation(market_data)
    market_data = apply_simpsons_rule(market_data)
    run_ml_clustering(market_data)
    run_supervised_ml(market_data)
    print("Pipeline complete. EDA figures saved to the 'figures' directory.")
