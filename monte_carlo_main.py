import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
assets = ['Stocks', 'Bonds']

portfolio_df = pd.DataFrame({
    'Asset': assets,
    'Weight': [0.6, 0.4],
    'Mean_Return': [0.07, 0.03],
    'Volatility': [0.15, 0.05]
})

# Correlation (not used for now because we're doing independent fat tails)
correlation = 0.1

# Fees
annual_fee = 0.01  # 1% management fee

# Fat tail settings
df_t = 5  # degrees of freedom for Student's t (fat tails!)

# Simulation settings
years = 20
n_simulations = 5000
def monte_carlo_fat_tails(portfolio_df, years, n_simulations, df_t, annual_fee, scenario_name='Normal'):
    n_assets = len(portfolio_df)
    mean_returns = portfolio_df['Mean_Return'].values
    volatilities = portfolio_df['Volatility'].values
    weights = portfolio_df['Weight'].values

    # Store all simulations
    simulation_results = np.zeros((years, n_simulations))

    for sim in range(n_simulations):
        portfolio_growth = 1.0  # Start at 1.0 (normalized)
        for year in range(years):
            # 1️ FAT TAIL: Simulate t-distributed returns (independent assets)
            t_randoms = np.random.standard_t(df_t, size=n_assets)
            simulated_returns = mean_returns + volatilities * t_randoms

            # 2️ Rebalance: Weighted portfolio return (weights re-applied each year)
            portfolio_return = np.dot(weights, simulated_returns)

            # 3️ Apply return + fees
            portfolio_growth *= (1 + portfolio_return)
            portfolio_growth *= (1 - annual_fee)  # Management fee

            simulation_results[year, sim] = portfolio_growth

    results_df = pd.DataFrame(simulation_results, index=range(1, years+1))
    results_df.columns = [f'{scenario_name}_Sim_{i}' for i in range(n_simulations)]
    return results_df
results_normal = monte_carlo_fat_tails(
    portfolio_df, years, n_simulations, df_t, annual_fee, scenario_name='Normal'
)
portfolio_crash = portfolio_df.copy()
portfolio_crash['Mean_Return'] = [0.05, 0.02]
portfolio_crash['Volatility'] = [0.25, 0.08]

results_crash = monte_carlo_fat_tails(
    portfolio_crash, years, n_simulations, df_t, annual_fee, scenario_name='Crash'
)

def plot_median_band(results_df, scenario_name):
    median = results_df.median(axis=1)
    p10 = results_df.quantile(0.1, axis=1)
    p90 = results_df.quantile(0.9, axis=1)

    plt.figure(figsize=(12,6))
    plt.plot(median, label=f'{scenario_name} Median', color='blue')
    plt.fill_between(median.index, p10, p90, color='lightblue', alpha=0.5, label='10%-90%')
    plt.title(f'Monte Carlo Simulation ({scenario_name}): Median & Confidence Band')
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

plot_median_band(results_normal, 'Normal')
plot_median_band(results_crash, 'Crash')
final_normal = results_normal.iloc[-1]
final_crash = results_crash.iloc[-1]

plt.figure(figsize=(10,6))
sns.kdeplot(final_normal, label='Normal Market', fill=True)
sns.kdeplot(final_crash, label='Crash Market', fill=True, color='red')
plt.title('Distribution of Final Portfolio Value: Normal vs. Crash Market')
plt.xlabel('Portfolio Value')
plt.legend()
plt.show()
def print_stats(final_values, scenario_name):
    mean_return = final_values.mean()
    median_return = final_values.median()
    std_dev = final_values.std()
    VaR_5 = final_values.quantile(0.05)
    CVaR_5 = final_values[final_values <= VaR_5].mean()
    sharpe_ratio = (mean_return - 0.02) / std_dev  # Assuming 2% risk-free rate

    print(f"\n📊 {scenario_name} Scenario Stats:")
    print(f"Mean Final Value: {mean_return:.2f}")
    print(f"Median Final Value: {median_return:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"5% VaR: {VaR_5:.2f}")
    print(f"5% CVaR (Expected Shortfall): {CVaR_5:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print_stats(final_normal, 'Normal')
print_stats(final_crash, 'Crash')

if __name__ == "__main__":
    # 🚀 RUN SIMULATIONS

    print("Running Normal Market Simulation...")
    results_normal = monte_carlo_fat_tails(
        portfolio_df, years, n_simulations, df_t, annual_fee, scenario_name='Normal'
    )

    print("Running Crash Market Simulation...")
    # Set up crash scenario
    portfolio_crash = portfolio_df.copy()
    portfolio_crash['Mean_Return'] = [0.05, 0.02]
    portfolio_crash['Volatility'] = [0.25, 0.08]

    results_crash = monte_carlo_fat_tails(
        portfolio_crash, years, n_simulations, df_t, annual_fee, scenario_name='Crash'
    )

    # PLOT RESULTS
    plot_median_band(results_normal, 'Normal')
    plot_median_band(results_crash, 'Crash')

    final_normal = results_normal.iloc[-1]
    final_crash = results_crash.iloc[-1]

    plt.figure(figsize=(10,6))
    sns.kdeplot(final_normal, label='Normal Market', fill=True)
    sns.kdeplot(final_crash, label='Crash Market', fill=True, color='red')
    plt.title('Distribution of Final Portfolio Value: Normal vs. Crash Market')
    plt.xlabel('Portfolio Value')
    plt.legend()
    plt.show()

    # PRINT STATS
    print_stats(final_normal, 'Normal')
    print_stats(final_crash, 'Crash')
