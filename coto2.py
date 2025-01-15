import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('server_metricsr.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

df.plot(figsize=(12, 4))
plt.show()

result = sm.tsa.stattools.adfuller(df['metric'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print(f'Critical Values: {result[4]}')

model = sm.tsa.ARIMA(df['metric'], order=(1, 1, 1))
results = model.fit()

forecast = results.forecast(steps=30)

plt.plot(df.index, df['metric'], label='Original')
plt.plot(forecast.index, forecast.values, label='Forecast')
plt.legend()
plt.show()
