import streamlit as st
from strategy import *

st.set_page_config(layout="wide")
st.title("📈 Portfolio Optimization Dashboard")

# Sidebar Inputs
assets = st.sidebar.text_input("Enter ticker symbols (comma-separated)", "AAPL, MSFT, GOOG, AMZN, META")    
bench = st.sidebar.text_input("Enter benchmark ticker symbol", "SPY")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
strategy = st.sidebar.selectbox("Optimization Strategy", ["Minimum Variance", "Maximum Sharpe", "Equal Weight", 
                                                          "Black-Litterman (Max Sharpe)", "CVaR"])
alpha = 0.95
if strategy == "CVaR":
    alpha = st.sidebar.slider("CVaR Confidence Level", 0.9,0.99,0.95)
views = []
if strategy == "Black-Litterman (Max Sharpe)":
    view_types = st.sidebar.multiselect("Optional: Add Views (Black-Litterman)", ["Absolute", "Relative"])
    view_idx = 0
    if "Absolute" in view_types:
        st.sidebar.markdown("### Absolute Views")
        num_abs = st.sidebar.number_input("Number of Absolute Views", 0, 10, 0, key="num_abs")
        for i in range(num_abs):
            col1, col2, col3 = st.sidebar.columns([1.5, 1, 1])
            ticker = col1.text_input(f"Ticker (Abs View #{i+1})")
            expected_return = col2.number_input("Expected Return (%)", key=f"abs_ret{i}", step = 0.5)
            confidence = col3.slider("Confidence (%)", 1, 100, 50, key=f"abs_conf{i}")
            if ticker:
                views.append(("absolute", ticker.upper(), expected_return / 100, confidence / 100))
    if "Relative" in view_types:
        st.sidebar.markdown("### Relative Views")
        num_rel = st.sidebar.number_input("Number of Relative Views", 0, 10, 0, key="num_rel")
        for i in range(num_rel):
            col1, col2, col3, col4 = st.sidebar.columns([1.5, 1.5, 1, 1])
            ticker1 = col1.text_input(f"Ticker A (#{i+1})")
            ticker2 = col2.text_input(f"Ticker B (#{i+1})")
            expected_diff = col3.number_input("A - B (%)", key=f"rel_diff{i}", step = 0.5)
            confidence = col4.slider("Confidence (%)", 1, 100, 50, key=f"rel_conf{i}")
            if ticker1 and ticker2:
                views.append(("relative", ticker1.upper(), ticker2.upper(), expected_diff / 100, confidence / 100))


@st.cache_data
def load_data_cached(tickers, start, end):
    return load_data(tickers, start, end)

if st.sidebar.button("Optimize Portfolio"):
    tickers = [ticker.strip().upper() for ticker in assets.split(",")]
    data = load_data_cached(tickers, start_date, end_date)
    benchmark = load_data_cached(bench, start_date, end_date)

    if data is not None:
        opt_weights, stats = optimize_portfolio(data, strategy, benchmark, views=views, alpha=alpha)

        st.subheader("Optimal Portfolio Weights")
        st.bar_chart(opt_weights)

        st.subheader("Portfolio Metrics (Annualized)")
        cols = st.columns(len(stats))
        for (k, v), col in zip(stats.items(), cols):
            with col:
                if k == 'Expected Return' or k == 'Sharpe Ratio':
                    st.metric(
                        label=k,
                        value=f"{v[0]:.4f}",
                        delta=f"{v[0]-v[1]:.4f}"
                        )               
                else:
                    st.metric(
                        label=k,
                        value=f"{v[0]:.4f}",
                        delta=f"{v[0]-v[1]:.4f}",
                        delta_color='inverse'
                        ) 
                    
        st.subheader("Portfolio Cumulative Returns (%)")
        st.line_chart(plot_historic_perf(data, opt_weights, benchmark))

        drawdown_data, stats = plot_drawdowns(data, opt_weights, benchmark)            
        st.subheader("Drawdown Analysis (%)")
        col1, col2 = st.columns(([3,1]))
        with col1:
            st.line_chart(drawdown_data * 100) 
        with col2:
            for (k, v) in stats.items():
                st.metric(label=k, value=f"{v:.2f}" if k!="Maximum Drawdown duration (days)" else f"{v:.0f}")
        
        st.subheader("Asset Correlation Heatmap")
        st.table(plot_correlation_heatmap(data))
    else:
        st.error("Failed to load data. Please check the ticker symbols and try again.")
