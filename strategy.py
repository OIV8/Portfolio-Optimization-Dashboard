import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, linprog
import seaborn as sns

def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)['Close']
        data = data.dropna(how='all') 
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def optimize_portfolio(data, strategy, benchmark, views=None, alpha=0.95):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    T, N = returns.shape

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_mean = benchmark_returns.mean()
    benchmark_var = benchmark_returns.var()

    if strategy == "Equal Weight":
        weights = np.array(N * [1/N])
    elif strategy == "Minimum Variance":
        def portfolio_std(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(N))
        initial_guess = N * [1/N]

        result = minimize(portfolio_std, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    elif strategy == "Maximum Sharpe":
        def neg_sharpe_ratio(weights):
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            daily_rf = 0.045/252
            return -1 * (port_return - daily_rf)/port_std

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(N))
        initial_guess = N * [1/N]

        result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    elif strategy == "Black-Litterman (Max Sharpe)":
        def black_litterman_expected_returns(tau, cov_matrix, confidences, P, Q, PI):
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
                        row = [0] * N
                        idx = data.columns.get_loc(ticker)
                        row[idx] = 1
                        P.append(row)
                        Q.append(expected_ret)
                        confidences.append(conf)
                elif view[0] == "relative":
                    _, ticker_a, ticker_b, expected_diff, conf = view
                    if ticker_a in data.columns and ticker_b in data.columns:
                        row = [0] * N
                        idx_a = data.columns.get_loc(ticker_a)
                        idx_b = data.columns.get_loc(ticker_b)
                        row[idx_a] = 1
                        row[idx_b] = -1
                        P.append(row)
                        Q.append(expected_diff)
                        confidences.append(conf)
            P = np.array(P)
            Q = np.array(Q)
            bl_returns = black_litterman_expected_returns(tau, cov_matrix, confidences, P, Q, PI)
        else:
            bl_returns = PI  

        def portfolio_std(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def neg_sharpe(weights):
            daily_rf = 0.045/252
            return -(np.dot(weights, bl_returns - daily_rf) / portfolio_std(weights))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(N))
        initial_guess = market_weights

        result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
    elif strategy=="CVaR":
        losses = -returns.values 
        
        # Variables: [weights (N), VaR (1), auxiliary (T)]
        # Objective: Minimize CVaR = VaR + (1/(1-alpha)) * mean(auxiliary)
        vars = N + 1 + T
        c = np.concatenate([np.zeros(N), [1], np.ones(T) / ((1 - alpha) * T)])

        # Equality Constraint: sum(weights) = 1
        A_eq = np.zeros((1,vars))
        A_eq[0:, :N] = 1
        b_eq = np.array([1])
        
        # Inequality Constraints:
        # 1. auxiliary >= 0
        A_ub1 = np.zeros((T, vars))  
        A_ub1[:, N+1:] = -np.eye(T)
        b_ub1 = np.zeros(T)
        # 2. auxiliary >= losses - VaR        
        A_ub2 = np.zeros((T, vars))  
        A_ub2[:, :N] = losses
        A_ub2[:, N] = -1 
        A_ub2[:, N+1:] = -np.eye(T)
        b_ub2 = np.zeros(T)
        
        A_ub = np.vstack([A_ub1, A_ub2])
        b_ub = np.hstack([b_ub1, b_ub2])
        
        # Bounds: weights >= 0, VaR unrestricted, auxiliary >= 0
        bounds = [(0, None)] * N + [(None, None)] + [(0, None)] * T
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        weights = result.x[:N]        
    else:
        raise ValueError("Invalid strategy")

    portfolio = pd.Series(weights, index=data.columns)
    portfolio_returns = returns.dot(weights)  
    sorted_returns = np.sort(portfolio_returns)
    var_hist = -np.percentile(sorted_returns, 100 * (1 - alpha))
    cvar_hist = -np.mean(sorted_returns[sorted_returns<=-var_hist])
    sorted_returns_bench = np.sort(benchmark_returns)
    var_hist_bench = -np.percentile(sorted_returns_bench, 100 * (1 - alpha))
    cvar_hist_bench = -np.mean(sorted_returns_bench[sorted_returns_bench<=-var_hist_bench])
    stats = {
        "Expected Return": [np.dot(weights, mean_returns) * 252, np.mean(benchmark_returns) * 252],
        "Volatility": [np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252) , 
                       np.std(benchmark_returns.to_numpy())* np.sqrt(252)],
        "Sharpe Ratio": [(np.dot(weights, mean_returns) * 252) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252),
                          np.mean(benchmark_returns) * 252/(np.std(benchmark_returns.to_numpy()) * np.sqrt(252))],
        f"VaR {alpha * 100:.0f}%" : [var_hist * np.sqrt(252), var_hist_bench * np.sqrt(252)],
        f"CVaR {alpha * 100:.0f}%" : [cvar_hist * np.sqrt(252), cvar_hist_bench * np.sqrt(252)]
    }

    return portfolio, stats


def plot_historic_perf(data, opt_weights, benchmark):    
    returns = data.pct_change().dropna()
    portfolio_returns = (returns *  opt_weights).sum(1)
    portfolio_perf = ((1 + portfolio_returns).cumprod() - 1) * 100
    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_perf = ((1 + benchmark_returns).cumprod() - 1) * 100

    historic_data = pd.concat([portfolio_perf, benchmark_perf], axis = 1)
    historic_data.columns = ['Portfolio', list(benchmark.columns)[0]]
    return historic_data    


def plot_drawdowns(data, opt_weights, benchmark):
    returns = data.pct_change().dropna()
    portfolio_returns = (returns *  opt_weights).sum(1)
    benchmark_returns = benchmark.pct_change().dropna()

    def calculate_drawdowns(returns):
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns
    
    portfolio_dd = calculate_drawdowns(portfolio_returns)
    benchmark_dd = calculate_drawdowns(benchmark_returns)
    drawdown_data = pd.concat([portfolio_dd, benchmark_dd], axis=1)
    drawdown_data.columns = ['Portfolio', list(benchmark.columns)[0]]
    
    stats = {
        'Portfolio Max Drawdown': drawdown_data['Portfolio'].min() * 100,
        'Benchmark Max Drawdown': drawdown_data[list(benchmark.columns)[0]].min() * 100,
        'Maximum Drawdown duration (days)': (drawdown_data['Portfolio'] == 0).astype(int).groupby(
        (drawdown_data['Portfolio'] == 0).astype(int).cumsum()).cumcount().max()
    }
    return drawdown_data, stats

def plot_correlation_heatmap(data):
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    cm = sns.dark_palette("blue", as_cmap=True)
    corr_matrix = corr_matrix.style.format(precision = 3).background_gradient(cmap=cm)
    return corr_matrix

