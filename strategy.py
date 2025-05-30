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
    portfolio_returns = (returns *  opt_weights).sum(1)
    portfolio_perf = ((1 + portfolio_returns).cumprod() - 1) * 100
    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_perf = ((1 + benchmark_returns).cumprod() - 1) * 100

    historic_data = pd.concat([portfolio_perf, benchmark_perf], axis = 1)
    historic_data.columns = ['Portfolio', list(benchmark.columns)[0]]
    return historic_data    


def plot_drawdowns(data, opt_weights, benchmark):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(opt_weights)
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

#######################
### Risks Functions ###
#######################
from copulae import StudentCopula
from scipy.stats import t
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def monte_carlo_simulation(data, opt_weights, num_days, num_simulations, initial_value, market_shocks, stock):
    num_stocks = len(opt_weights)
    log_returns = np.log(data / data.shift(1)).dropna()
    garch_models = {}
    standardized_residuals = np.zeros((len(log_returns), num_stocks))
    conditional_volatilities = np.zeros((len(log_returns), num_stocks))
    
    for i in range(num_stocks):
        col_name = log_returns.columns[i]
        model = arch_model(
            log_returns.iloc[:, i], 
            mean='Zero', 
            vol='GARCH', 
            p=1, q=1, 
            dist='studentst',
            rescale=False,
        ).fit(disp='off')
        
        garch_models[col_name] = model
        standardized_residuals[:, i] = model.std_resid.values
        conditional_volatilities[:, i] = model.conditional_volatility.values
    
    uniform_data = np.zeros_like(standardized_residuals)
    for i in range(num_stocks):
        col_name = log_returns.columns[i]
        df_param = garch_models[col_name].params['nu']
        uniform_data[:, i] = t.cdf(standardized_residuals[:, i], df=df_param)
    
    student_copula = StudentCopula(dim=num_stocks)
    student_copula.fit(uniform_data)

    forecasted_volatilities = np.zeros((num_days, num_stocks))
    mean_returns = log_returns.mean().values
    
    for i in range(num_stocks):
        col_name = log_returns.columns[i]
        model = garch_models[col_name]
        
        # Forecast volatilities
        forecasts = model.forecast(horizon=num_days, reindex=False)
        forecasted_volatilities[:, i] = np.sqrt(forecasts.variance.values[-1])
    
    ### Monte Carlo simulation
    simulated_portfolio_paths = np.zeros((num_simulations, num_days + 1))
    simulated_portfolio_paths[:, 0] = initial_value
    final_portfolio_values = np.zeros(num_simulations)
    total_portfolio_returns = np.zeros(num_simulations)
    
    for s in range(num_simulations):

        uniform_samples = student_copula.random(num_days)
        standardized_innovations = np.zeros((num_days, num_stocks))
        for i in range(num_stocks):
            col_name = log_returns.columns[i]
            df_param = garch_models[col_name].params['nu']
            standardized_innovations[:, i] = t.ppf(uniform_samples[:, i], df=df_param)
        
        daily_log_returns = np.zeros((num_days, num_stocks))
        for day in range(num_days):
            for i in range(num_stocks):
                daily_log_returns[day, i] = (
                    mean_returns[i] + 
                    forecasted_volatilities[day, i] * standardized_innovations[day, i]
                )
        
        daily_arithmetic_returns = np.exp(daily_log_returns) - 1
        daily_portfolio_returns = np.dot(daily_arithmetic_returns, opt_weights)
        
        for shock_day, shock_magnitude in market_shocks:
            daily_portfolio_returns[shock_day] += shock_magnitude
        
        portfolio_value_path = np.zeros(num_days + 1)
        portfolio_value_path[0] = initial_value
        
        for day in range(num_days):
            portfolio_value_path[day + 1] = (
                portfolio_value_path[day] * (1 + daily_portfolio_returns[day])
            )
        
        simulated_portfolio_paths[s, :] = portfolio_value_path
        final_portfolio_values[s] = portfolio_value_path[-1]
        total_portfolio_returns[s] = (final_portfolio_values[s] - initial_value) / initial_value
    
        
    # Plot Monte Carlo Results
    figure = make_subplots(rows=1, cols=2,
                            subplot_titles=('Portfolio Value Paths 100 Samples)','Total Value Distribution'),
                            shared_yaxes=True, horizontal_spacing = 0, column_widths=[3,1])
    sample_paths = simulated_portfolio_paths[:100] 
    time_ax = pd.date_range(start=data.index[-1] , periods=num_days + 1, freq='B')
    for i, path in enumerate(sample_paths):
        figure.add_trace(go.Scatter(x=time_ax, y=path, mode='lines', 
                    showlegend=False, opacity=0.3),
                    row = 1, col=1)
    figure.add_trace(go.Histogram(y=final_portfolio_values[:100],orientation='h', nbinsy=25, 
                    name='Final Values', showlegend=False ),
                        row=1, col=2)
    figure.update_layout(height=700)

    # Plot copula correlation 
    asset_i = data.columns.get_loc(stock)
    asset_names = data.columns.tolist()
    stocks_indices = [i for i in range(num_stocks) if i != asset_i]
    fig = make_subplots(
            rows=1, cols=num_stocks-1,
            subplot_titles=[f'{asset_names[asset_i]} vs {asset_names[i]}' for i in stocks_indices])
    for j, asset_j in enumerate(stocks_indices):
        u1, u2 = uniform_data[:, asset_i], uniform_data[:, asset_j]
        lower_tail = (u1 <= 0.1) & (u2 <= 0.1)
        upper_tail = (u1 >= 0.9) & (u2 >= 0.9)
        normal_region = ~(lower_tail | upper_tail)
        
        fig.add_trace(
            go.Scatter(
                x=u1[normal_region], y=u2[normal_region],
                mode='markers',
                marker=dict(size=3, color='gray', opacity=0.4),
                hovertemplate=f'{asset_names[asset_i]}: %{{x:.3f}}<br>{asset_names[asset_j]}: %{{y:.3f}}<extra></extra>',
            ),
            row=1, col=j+1
        )
        
        if np.any(lower_tail):
            fig.add_trace(
                go.Scatter(
                    x=u1[lower_tail], y=u2[lower_tail],
                    mode='markers',
                    marker=dict(size=5, color='red', opacity=0.8),
                    hovertemplate=f'{asset_names[asset_i]}: %{{x:.3f}}<br>{asset_names[asset_j]}: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=j+1
            )
        
        if np.any(upper_tail):
            fig.add_trace(
                go.Scatter(
                    x=u1[upper_tail], y=u2[upper_tail],
                    mode='markers',
                    marker=dict(size=5, color='blue', opacity=0.8),
                    hovertemplate=f'{asset_names[asset_i]}: %{{x:.3f}}<br>{asset_names[asset_j]}: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=j+1
            )
    fig.update_layout(title_text = 'Copula Correlations',showlegend=False)
   
    stats = {'Expected Return': total_portfolio_returns.mean(),
                'Volatility': total_portfolio_returns.std(),
                'Probability of loss': (total_portfolio_returns < 0).sum()/num_simulations,
                'VaR 95%': -np.percentile(total_portfolio_returns, 5),
                'CVaR 95%': -np.mean(total_portfolio_returns[total_portfolio_returns <= np.percentile(total_portfolio_returns, 5)])}
    
    return stats, figure, fig

