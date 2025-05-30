# 📈 Portfolio Optimization Dashboard
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An **interactive dashboard** for portfolio optimization strategies — from classic mean-variance to Black-Litterman with custom investor views. Features a robust risk analysis module that performs Monte Carlo simulations based on a Student's t-Copula (calibrated using GARCH(1,1)-t residuals for each asset). This approach help to models non-linear dependencies and heavy tails in asset returns, enabling realistic stress testing under custom market shocks and providing more accurate VaR and CVaR estimations.

## Features

- Portfolio optimization strategies:
  - Minimum Variance
  - Maximum Sharpe Ratio
  - Equal Weight
  - **Black-Litterman** (with custom absolute & relative views + confidence levels)
  - **CVaR** (Rockafellar and Ursayev (2001))

- Investor view inputs (for Black-Litterman and CVaR):
  - Absolute views (e.g., "AAPL will return 10%")
  - Relative views (e.g., "AAPL will outperform MSFT by 3%")
  - Adjustable confidence levels for each view and for CVaR

- Risk Analysis:
  - **Monte Carlo Simulation** for probabilistic forecasting of portfolio performance.
  - **Student's t-Copula integration** (calibrated with GARCH(1,1)-t residuals).
  - Calculation of **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** for comprehensive downside risk assessment.
  - **Stress testing** through **customizable market shocks** (e.g., flash crashes).


- 📆 Historical portfolio performance plot
- 📎 Benchmark comparison (e.g., SPY)
- 🌐 Fully interactive via [Streamlit](https://streamlit.io/)


##  🌐 Web App
To run the app:

```bash
streamlit run stream.py
```

## 📸 Screenshots

<img src="tab_optimize.png" width="800">
<img src="tab_risk.png" width="800">


## 💡 To-Do 

- [ ] Factor Exposure
- [ ] Risk contribution
- [ ] Sector / ESG filtering


## 📁 File structure
- `strategy.py`- Retrieve data and implement optimization strategy.
- `stream.py`- Builds an interactive web app using Streamlit to visualize the dashboard.
- `screenshots` - Screenshots folder
  - `tab_optimize.png`
  - `tab_risk.png`
- `README.md`- Project documentation.

