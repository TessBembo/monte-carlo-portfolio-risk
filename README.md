# Monte Carlo Portfolio Risk Simulation

This project runs an enhanced Monte Carlo simulation of a 60/40 portfolio with:

-  Fat-tailed risk modeling (Student’s t-distribution)
-  Dynamic rebalancing
-  Management fees/slippage
-  Stress-test scenario comparison (normal vs. crash-heavy)
-  Advanced metrics: VaR, CVaR, Sharpe Ratio

## Features

- Multi-year simulation (20 years)
- Side-by-side scenario analysis
- Beautiful plots 

## Preview

### Time Evolution Plot (Normal Market)

![Normal Market Median Band](normal_market_median_band.png)

### Distribution Comparison: Normal vs. Crash

![Distribution Comparison](distribution_comparison.png)

## How to Run

1 Install requirements:

```bash
pip install numpy pandas matplotlib seaborn

2 Run the simulation:

python monte_carlo_main.py

**Full write-up & results here:** [My Blog Post](TessBembo.github.io)
