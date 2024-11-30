import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta




# Varsayılan stili kullanmak için
plt.style.use('default')

class MACDBacktester():
    def __init__(self, symbol, EMA_S, EMA_L, signal_mw, start, end, tc):
        self.symbol = symbol
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_mw = signal_mw
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None 
        self.get_data()
        
    def get_data(self):
        try:
            raw = yf.download(tickers=self.symbol, start=self.start, end=self.end, interval='1d')
            
            if isinstance(raw, pd.DataFrame):
                if 'Adj Close' in raw.columns:
                    raw = raw['Adj Close']
            
            raw = pd.DataFrame(raw)
            raw.columns = ['price'] if len(raw.columns) > 0 else ['price']
            
            raw = raw.dropna()
            raw["returns"] = np.log(raw / raw.shift(1))
            raw["EMA_S"] = raw["price"].ewm(span=self.EMA_S, min_periods=self.EMA_S).mean() 
            raw["EMA_L"] = raw["price"].ewm(span=self.EMA_L, min_periods=self.EMA_L).mean()
            raw["MACD"] = raw.EMA_S - raw.EMA_L
            raw["MACD_Signal"] = raw.MACD.ewm(span=self.signal_mw, min_periods=self.signal_mw).mean() 
            self.data = raw
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {e}")
            self.data = None

    def test_strategy(self):
        if self.data is None:
            return None
        
        data = self.data.copy().dropna()
        data["position"] = np.where(data["MACD"] > data["MACD_Signal"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy - data.trades * self.tc
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        if len(data) == 0:
            return None
        
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def list_last_buy_signals(self):
        if self.results is None or len(self.results) < 2:
            return None

        if (self.results["MACD"].iloc[-2] < self.results["MACD_Signal"].iloc[-2]) and \
           (self.results["MACD"].iloc[-1] > self.results["MACD_Signal"].iloc[-1]):
            return self.results[["price", "MACD", "MACD_Signal"]].iloc[-1]
        else:
            return None

def plot_macd_strategy(data):
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    plt.subplot(2, 1, 1)
    plt.title('Price and Buy/Sell Signals', fontsize=15, fontweight='bold')
    plt.plot(data.index, data['price'], label='Price', color='blue')
    
    buy_signals = data[data['position'] == 1]
    sell_signals = data[data['position'] == -1]
    plt.scatter(buy_signals.index, buy_signals['price'], color='green', label='Buy', marker='^', s=100)
    plt.scatter(sell_signals.index, sell_signals['price'], color='red', label='Sell', marker='v', s=100)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title('MACD and Signal Line', fontsize=15, fontweight='bold')
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['MACD_Signal'], label='Signal Line', color='red')
    plt.legend()
    
    plt.tight_layout()
    return plt

def analyze_all_stocks(bist_symbols, ema_short, ema_long, signal_mw, start, end, transaction_cost):
    buy_signals = []
    progress_bar = st.progress(0)
    for i, symbol in enumerate(bist_symbols):
        try:
            macd_backtester = MACDBacktester(
                symbol=symbol,
                EMA_S=ema_short,
                EMA_L=ema_long,
                signal_mw=signal_mw,
                start=start,
                end=end,
                tc=transaction_cost
            )
            result = macd_backtester.test_strategy()
            if result is not None:
                last_signal = macd_backtester.list_last_buy_signals()
                if last_signal is not None:
                    buy_signals.append((symbol, last_signal["price"]))
            
            progress_bar.progress((i + 1) / len(bist_symbols))
        except Exception as e:
            st.warning(f"Error analyzing {symbol}: {e}")
    
    progress_bar.empty()
    return buy_signals

def main():
    st.set_page_config(
        page_title="MACD Backtester",
        page_icon="📈",
        layout="wide"
    )

    st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    bist_100_symbols = [
        # BIST 100 sembolleri buraya gelecek
        'AEFES.IS', 'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 
        'DOAS.IS', 'ENJSA.IS', 'GOZDE.IS', 'HEKTS.IS', 'KCHOL.IS', 
        'KOZAA.IS', 'MGROS.IS', 'PETKM.IS', 'SAHOL.IS', 'SASA.IS', 
        'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS',
        'TTKOM.IS', 'TTRAK.IS', 'TUPRS.IS', 'ULKER.IS', 'YKBNK.IS'
    ]

    st.title("📈 BIST 100 MACD Backtester")

    st.sidebar.header("Strategy Parameters")
    
    selected_symbol = st.sidebar.selectbox(
        "Select Stock", 
        bist_100_symbols, 
        help="Choose a stock from BIST 100 index to analyze"
    )

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        ema_short = st.number_input('Short EMA', value=12, min_value=1, help='Shorter EMA period')
    with col2:
        ema_long = st.number_input('Long EMA', value=26, min_value=1, help='Longer EMA period')
    with col3:
        signal_mw = st.number_input('Signal MW', value=9, min_value=1, help='Signal line period')

    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())

    transaction_cost = st.sidebar.slider(
        'Transaction Cost (%)', 
        min_value=0.0, 
        max_value=1.0, 
        value=0.001, 
        step=0.0001,
        help='Cost per trade as a percentage'
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        backtest_btn = st.sidebar.button('Run Backtest 🚀')
    with col2:
        analyze_all_btn = st.sidebar.button('Analyze All Stocks 🔍')

    if backtest_btn:
        with st.spinner('Running Backtest...'):
            macd_backtester = MACDBacktester(
                symbol=selected_symbol,
                EMA_S=ema_short,
                EMA_L=ema_long,
                signal_mw=signal_mw,
                start=start_date,
                end=end_date,
                tc=transaction_cost
            )

            result = macd_backtester.test_strategy()

            if result:
                strategy_performance, out_performance = result
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Strategy Performance", f"{strategy_performance:.2f}", delta=f"{out_performance:.2f}")
                with col2:
                    st.metric("Total Return", f"{(strategy_performance-1)*100:.2f}%")
                
                fig = plot_macd_strategy(macd_backtester.results)
                st.pyplot(fig)
                
                last_buy_signal = macd_backtester.list_last_buy_signals()
                if last_buy_signal is not None:
                    st.subheader('Last Buy Signal')
                    st.write(last_buy_signal)
            else:
                st.warning('No data available for the selected parameters.')

    if analyze_all_btn:
        st.subheader('Analyzing All Stocks...')
        buy_signals = analyze_all_stocks(
            bist_symbols=bist_100_symbols,
            ema_short=ema_short,
            ema_long=ema_long,
            signal_mw=signal_mw,
            start=start_date,
            end=end_date,
            transaction_cost=transaction_cost
        )
        if buy_signals:
            st.subheader('Stocks with Buy Signals')
            df = pd.DataFrame(buy_signals, columns=['Symbol', 'Price'])
            st.dataframe(df)
        else:
            st.info('No Buy signals found for the given parameters.')

if __name__ == '__main__':
    main()
