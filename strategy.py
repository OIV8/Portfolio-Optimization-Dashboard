import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)['Close']
        data = data.dropna(how='all') 
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def optimize_portfolio(data, strategy, benchmark, views=None):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_mean = benchmark_returns.mean()
    benchmark_var = benchmark_returns.var()

    if strategy == "Equal Weight":
        weights = np.array(num_assets * [1/num_assets])
    elif strategy == "Minimum Variance":
        def portfolio_std(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1/num_assets]

        result = minimize(portfolio_std, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    elif strategy == "Maximum Sharpe":
        def neg_sharpe_ratio(weights):
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            daily_rf = 0.045/252
            return -1 * (port_return - daily_rf)/port_std

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1/num_assets]

        result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    elif strategy == "Black-Litterman (Max Sharpe)":
        def black_litterman_expected_returns(tau, cov_matrix, P, Q, PI):
            omega = np.zeros((len(Q), len(Q)))
            for i, conf in enumerate(confidences):
                epsilon = 1e-8  # Small value to prevent division errors when confidence is 100
                base_var = np.dot(np.dot(P[i], tau * cov_matrix), P[i])
                omega[i, i] = max((1 - conf) * base_var, epsilon)
            left = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
            adjusted_returns = np.dot(left, np.dot(np.linalg.inv(tau * cov_matrix), PI) +  (np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)))
            return adjusted_returns
        
        market_cap = np.array([yf.Ticker(t).info["marketCap"] for t in data.columns])
        market_weights = market_cap / np.sum(market_cap)    
        tau = 0.05
        daily_rf = 0.045/252
        lambda_ = (benchmark_mean - daily_rf)/benchmark_var
        PI = lambda_.iloc[0] * np.dot(cov_matrix, market_weights)

        if views:
            P, Q, confidences = [], [], []
            for view in views:
                if view[0] == "absolute":
                    _, ticker, expected_ret, conf = view
                    if ticker in data.columns:
                        row = [0] * num_assets
                        idx = data.columns.get_loc(ticker)
                        row[idx] = 1
                        P.append(row)
                        Q.append(expected_ret)
                        confidences.append(conf)
                elif view[0] == "relative":
                    _, ticker_a, ticker_b, expected_diff, conf = view
                    if ticker_a in data.columns and ticker_b in data.columns:
                        row = [0] * num_assets
                        idx_a = data.columns.get_loc(ticker_a)
                        idx_b = data.columns.get_loc(ticker_b)
                        row[idx_a] = 1
                        row[idx_b] = -1
                        P.append(row)
                        Q.append(expected_diff)
                        confidences.append(conf)
            P = np.array(P)
            Q = np.array(Q)
            bl_returns = black_litterman_expected_returns(tau, cov_matrix, P, Q, PI)
        else:
            bl_returns = PI  

        def portfolio_std(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def neg_sharpe(weights):
            daily_rf = 0.045/252
            return -(np.dot(weights, bl_returns - daily_rf) / portfolio_std(weights))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = market_weights

        result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    else:
        raise ValueError("Invalid strategy")

    portfolio = pd.Series(weights, index=data.columns)
    stats = {
        "Expected Return": [np.dot(weights, mean_returns) * 252, np.mean(benchmark_returns) * 252],
        "Volatility": [np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252) , 
                       np.std(benchmark_returns.to_numpy())* np.sqrt(252)],
        "Sharpe Ratio": [(np.dot(weights, mean_returns) * 252) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252),
                          np.mean(benchmark_returns) * 252/(np.std(benchmark_returns.to_numpy()) * np.sqrt(252))]
    }

    return portfolio, stats


def plot_historic_perf(data, opt_weights, benchmark):    
    returns = data.pct_change().dropna()
    portfolio_ret = (returns *  opt_weights).sum(1)
    portfolio_perf = ((1 + portfolio_ret).cumprod() - 1) * 100

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_perf = ((1 + benchmark_returns).cumprod() - 1) * 100

    historic_data = pd.concat([portfolio_perf, benchmark_perf], axis = 1)
    historic_data.columns = ['Portfolio', 'S&P 500']
    
    return historic_data
    
