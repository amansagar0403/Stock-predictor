import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler
from datetime import datetime

stocks = [
    # U.S. Large-Cap Stocks
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'INTC', 'IBM',
    'BA', 'JPM', 'GS', 'WFC', 'C', 'PYPL', 'V', 'MA', 'KO', 'PEP', 'XOM', 'CVX', 'BP',
    'TSM', 'ADBE', 'CRM', 'ORCL', 'AMD', 'QCOM', 'UBER', 'DIS', 'T', 'TMUS', 'BABA',
    'PFE', 'JNJ', 'MRNA', 'GILD', 'UNH', 'LLY', 'NKE', 'MCD', 'SBUX', 'WMT', 'COST',
    'HD', 'LOW', 'TGT', 'CVS', 'ABBV', 'VRTX', 'REGN', 'DHR', 'MMM', 'GE', 'HON',
    'CAT', 'DE', 'UPS', 'FDX', 'CSCO', 'TXN', 'AVGO', 'LRCX', 'MU', 'ADI', 'NXPI',
    'SNPS', 'CDNS', 'KLAC', 'SPGI', 'MSCI', 'ICE', 'CME', 'MCO', 'AON', 'MMC', 'AJG',
    'PGR', 'TRV', 'EBAY', 'ROKU', 'SQ', 'TWLO', 'ZM', 'SHOP', 'DOCU', 'OKTA', 'TTD',
    'DDOG', 'FSLR', 'ENPH', 'SEDG', 'PLUG', 'RUN', 'BLDP', 'NKLA', 'FSR', 'QS', 'XPEV',
    'LI', 'NIO', 'RIVN', 'LCID', 'ASTS', 'SPCE', 'RKLB', 'ACHR', 'JOBY', 'EH', 'ABNB',
    'BKNG', 'EXPE', 'TRIP', 'MAR', 'HLT', 'H', 'WH', 'WYN', 'CHH', 'DAL', 'UAL', 'LUV',
    'AAL', 'RYAAY', 'SAVE', 'ALK', 'AZO', 'ORLY', 'AAP', 'GPC', 'F', 'GM', 'STLA', 'TM',
    'HMC', 'FCAU', 'NSANY', 'VOW3.DE', 'DAI.DE', 'BMW.DE', 'MBG.DE', 'AIR.PA', 'LHA.DE',
    'RACE', 'TT', 'CARR', 'JCI', 'LEN', 'DHI', 'PHM', 'NVR', 'TOL', 'KBH', 'HOV', 'BZH',
    'MAS', 'MLM', 'VMC', 'EXP', 'USCR', 'SUM', 'CX', 'X', 'STLD', 'NUE', 'MT', 'RS',
    'CMC', 'CLF', 'SCCO', 'FCX', 'BHP', 'RIO', 'VALE', 'GOLD', 'NEM', 'AEM', 'FNV',
    'WPM', 'AG', 'CDE', 'PAAS', 'HL', 'MUX', 'AU', 'SBGL', 'GFI', 'KGC', 'BTG', 'RGLD',

    # U.S. Mid-Cap and Small-Cap Stocks
    'ETSY', 'PTON', 'PLTR', 'SNOW', 'DASH', 'CRWD', 'ZS', 'NET', 'ASAN', 'UPST',
    'AFRM', 'COIN', 'HOOD', 'SOFI', 'RBLX', 'U', 'DKNG', 'CHWY', 'FVRR', 'APPS',
    'SPCE', 'RKT', 'WKHS', 'BLNK', 'CHPT', 'BEEM', 'PLUG', 'FCEL', 'BLDP', 'NKLA',
    'QS', 'HYLN', 'LAZR', 'VLDR', 'MVIS', 'AEVA', 'ARVL', 'GOEV', 'LEV', 'RIDE',
    'PSFE', 'CLOV', 'WISH', 'SDC', 'BB', 'NOK', 'AMC', 'GME', 'BBBY', 'KOSS',
    'EXPR', 'NAKD', 'SNDL', 'TLRY', 'ACB', 'CGC', 'CRON', 'HEXO', 'OGI', 'APHA',

    # International Stocks (Europe, Asia, Emerging Markets)
    'ASML', 'NVO', 'SAP', 'SAN', 'AZN', 'HSBC', 'UL', 'BP', 'RY', 'TD', 'BNS', 'BMO',
    'ENB', 'CNQ', 'SHOP', 'BIDU', 'JD', 'TCEHY', 'PDD', 'BABA', 'NTES', 'TCOM', 'YUMC',
    'BILI', 'IQ', 'NIO', 'XPEV', 'LI', 'BYDDF', 'TSM', 'UMC', 'ASX', 'HMC', 'TM', 'SONY',
    'NTT', 'MFG', 'SNE', 'INFY', 'TCS', 'HDB', 'SBIN', 'RELIANCE.NS', 'TATAMOTORS.NS',
    'MARUTI.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'ITC.NS', 'LT.NS', 'TATASTEEL.NS',

    # Crypto and Blockchain-Related Stocks
    'COIN', 'MARA', 'RIOT', 'SI', 'MSTR', 'HUT', 'BITF', 'BTBT', 'CLSK', 'ARBK',
    'CAN', 'HIVE', 'DMGI', 'GLXY.TO', 'BITI.TO', 'BLOK', 'BKCH', 'LEO', 'GBTC',

    # EV and Renewable Energy Stocks
    'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'FSR', 'WKHS', 'NKLA', 'GOEV',
    'BLNK', 'CHPT', 'BEEM', 'PLUG', 'FCEL', 'BLDP', 'RUN', 'ENPH', 'SEDG', 'FSLR',
    'SPWR', 'NOVA', 'ARRY', 'MAXN', 'CSIQ', 'JKS', 'DQ', 'SOL', 'SUNW', 'NEE', 'BEP',
    'BAM', 'ORA', 'CWEN', 'AY', 'NEP', 'HASI', 'TERP', 'AMRC', 'NOVA', 'STEM',

    # SPACs (Special Purpose Acquisition Companies)
    'IPOF', 'IPOD', 'PSTH', 'CCIV', 'SOAC', 'THCB', 'STPK', 'BFT', 'IPV', 'GHIV',
    'QS', 'LAZR', 'HYLN', 'RTP', 'PIC', 'BTWN', 'ACIC', 'FUSE', 'VGAC', 'SNPR',

    # Meme and High-Volatility Stocks
    'GME', 'AMC', 'BB', 'NOK', 'KOSS', 'EXPR', 'NAKD', 'SNDL', 'TLRY', 'BBBY',
    'CLOV', 'WISH', 'SDC', 'RKT', 'WKHS', 'SPCE', 'PLTR', 'CLNE', 'OCGN', 'CTRM',

    # Dividend Stocks
    'T', 'VZ', 'MO', 'PM', 'KO', 'PEP', 'PG', 'JNJ', 'MRK', 'PFE', 'ABBV', 'MMM',
    'XOM', 'CVX', 'BP', 'O', 'SPG', 'TGT', 'WMT', 'COST', 'HD', 'LOW', 'NKE', 'MCD',
    'SBUX', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHD', 'VYM', 'NOBL',

    # Healthcare and Biotech Stocks
    'MRNA', 'BNTX', 'PFE', 'JNJ', 'MRK', 'ABBV', 'GILD', 'REGN', 'VRTX', 'ILMN',
    'TDOC', 'CVS', 'WBA', 'CI', 'UNH', 'HUM', 'ANTM', 'DHR', 'ISRG', 'SYK', 'BAX',
    'BSX', 'ZTS', 'ALXN', 'BIIB', 'AMGN', 'VRTX', 'INCY', 'EXEL', 'SRPT', 'CRSP',
    'EDIT', 'BEAM', 'NTLA', 'VERV', 'RARE', 'BLUE', 'FATE', 'KPTI', 'KURA', 'SRRK',

    # Industrials and Defense Stocks
    'BA', 'LMT', 'RTX', 'NOC', 'GD', 'HII', 'LHX', 'TXT', 'CAT', 'DE', 'HON',
    'GE', 'MMM', 'EMR', 'ITW', 'ROK', 'SWK', 'FAST', 'NDSN', 'DOV', 'PH', 'AME',
    'ETN', 'ROP', 'TT', 'CARR', 'JCI', 'OTIS', 'IR', 'AOS', 'WAB', 'GNRC', 'FLR',

    # Consumer Goods and Retail Stocks
    'NKE', 'TGT', 'WMT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'YUM', 'CMG', 'DPZ',
    'DRI', 'MKC', 'K', 'GIS', 'CPB', 'SJM', 'HSY', 'KHC', 'MDLZ', 'CL', 'EL', 'PG',
    'KO', 'PEP', 'MNST', 'STZ', 'BF.B', 'TAP', 'SAM', 'FIZZ', 'COKE', 'BJ', 'CASY',

    # Financials and Insurance Stocks
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'COF', 'DFS',
    'V', 'MA', 'PYPL', 'SQ', 'SOFI', 'ALLY', 'AIG', 'MET', 'PRU', 'LNC', 'PFG',
    'HIG', 'ALL', 'TRV', 'PGR', 'AFL', 'BRO', 'RE', 'SPGI', 'MCO', 'NDAQ', 'CME',
    'ICE', 'CBOE', 'MKTX', 'TW', 'STT', 'NTRS', 'FITB', 'KEY', 'CFG', 'HBAN',

    # Real Estate and REITs
    'O', 'SPG', 'AMT', 'CCI', 'EQIX', 'DLR', 'PLD', 'WELL', 'VTR', 'AVB', 'EQR',
    'ESS', 'UDR', 'MAA', 'CPT', 'ARE', 'SBAC', 'PSA', 'EXR', 'LSI', 'REXR', 'FRT',
    'KIM', 'REG', 'BRX', 'PEAK', 'IRT', 'NHI', 'OHI', 'DOC', 'MPW', 'HR', 'STAG',

    # Utilities and Energy Infrastructure
    'NEE', 'DUK', 'D', 'SO', 'EXC', 'AEP', 'XEL', 'PEG', 'ED', 'EIX', 'FE', 'ETR',
    'AWK', 'WTRG', 'CNP', 'ATO', 'LNT', 'CMS', 'DTE', 'NI', 'PNW', 'SRE', 'OKE',
    'WMB', 'KMI', 'TRP', 'ENB', 'EPD', 'ET', 'MPLX', 'PAA', 'PBA', 'SUN', 'VLO',
    'PSX', 'MPC', 'HFC', 'DK', 'CVI', 'PBF', 'CLR', 'MRO', 'DVN', 'FANG', 'EOG',
    'COP', 'CVX', 'XOM', 'BP', 'RDS.A', 'RDS.B', 'TTE', 'EQNR', 'ENI', 'REPYY',
]

def get_stock_data(ticker, start='2015-01-01'):
    """Download historical stock data for a given ticker."""
    try:
        end = datetime.today().strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"No data found for {ticker}")
            return None
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    try:
        # Ensure we're working with Series
        close_price = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Calculate technical indicators
        rsi = ta.momentum.RSIIndicator(close_price)
        macd = ta.trend.MACD(close_price)
        bb = ta.volatility.BollingerBands(close_price)
        volume_sma = ta.trend.SMAIndicator(volume, window=20)  # Changed from ta.volume to ta.trend
        sma20 = ta.trend.SMAIndicator(close_price, window=20)
        sma50 = ta.trend.SMAIndicator(close_price, window=50)
        
        # Add indicators to dataframe
        df['RSI'] = rsi.rsi()
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['Volume_SMA'] = volume_sma.sma_indicator()
        df['SMA_20'] = sma20.sma_indicator()
        df['SMA_50'] = sma50.sma_indicator()
        
        # Add additional basic technical indicators
        df['Daily_Return'] = close_price.pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        return df
    except Exception as e:
        print(f"Error calculating technical indicators for {df['Ticker'].iloc[0]}: {e}")
        return None

def preprocess_data(df):
    """Preprocess and normalize the data."""
    try:
        # Drop rows with missing values
        df = df.dropna()
        
        # Select features for scaling (excluding 'Close' and 'Volume')
        features_to_scale = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 
                           'Volume_SMA', 'SMA_20', 'SMA_50', 'Volatility']
        
        # Create a copy of the features to scale
        scale_df = df[features_to_scale].copy()
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Scale features
        scaled_features = scaler.fit_transform(scale_df)
        
        # Create new dataframe with scaled features
        df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)
        
        # Add back non-scaled columns
        df_scaled['Date'] = df['Date'].values
        df_scaled['Ticker'] = df['Ticker'].values
        df_scaled['Close'] = df['Close'].values
        df_scaled['Volume'] = df['Volume'].values
        df_scaled['Daily_Return'] = df['Daily_Return'].values
        
        # Reorder columns
        cols = ['Date', 'Ticker', 'Close'] + features_to_scale + ['Volume', 'Daily_Return']
        df_scaled = df_scaled[cols]
        
        return df_scaled
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def process_stocks(stocks, start_date='2010-01-01', batch_size=50):
    """Process stocks in batches and combine the results."""
    all_data = []
    total_batches = (len(stocks) + batch_size - 1) // batch_size
    
    # Process stocks in batches to avoid memory issues
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{total_batches}")
        
        for stock in batch:
            print(f"Processing {stock}...", end=' ')
            try:
                # Get stock data
                df = get_stock_data(stock, start_date)
                if df is None:
                    continue
                
                # Add technical indicators
                df = add_technical_indicators(df)
                if df is None:
                    continue
                
                # Preprocess data
                df = preprocess_data(df)
                if df is None:
                    continue
                
                all_data.append(df)
                print("Done!")
            except Exception as e:
                print(f"Failed: {e}")
                continue
    
    # Combine all processed data
    if all_data:
        final_dataset = pd.concat(all_data, ignore_index=True)
        return final_dataset
    else:
        return None

def save_to_csv(df, filename='stocks_dataset.csv'):
    """Save the processed dataset to a CSV file."""
    try:
        df.to_csv(filename, index=False)
        print(f"\nDataset successfully saved to {filename}")
        print(f"File size: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Error saving dataset: {e}")

# Main execution
if __name__ == "__main__":
    # Process stocks in batches
    dataset = process_stocks(stocks, batch_size=50)
    
    if dataset is not None:
        # Save the dataset
        save_to_csv(dataset)
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total number of records: {len(dataset):,}")
        print(f"Number of unique stocks: {dataset['Ticker'].nunique()}")
        print(f"Date range: {dataset['Date'].min()} to {dataset['Date'].max()}")
        print("\nSample of the dataset:")
        print(dataset.head())
        
        # Print feature statistics
        print("\nFeature Statistics:")
        print(dataset.describe().round(2))
    else:
        print("No data was processed successfully.")