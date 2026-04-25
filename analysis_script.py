import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
import datetime
import os

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

if __name__ == "__main__":
    market_data, event_data = load_and_prep_data()
    run_eda(market_data)
    run_hypothesis_test(market_data)
    print_top_10_spikes(market_data, event_data)
    print("Pipeline complete. EDA figures saved to the 'figures' directory.")
