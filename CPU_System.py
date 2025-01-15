import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df6 = pd.read_csv("cpu_DEVELOP.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df6["System_pct"] = (df6["System"]/df6["Core"])
df6["z_score"] = stats.zscore(df6["System_pct"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df6["anomaly_z"] = df6["z_score"].apply(lambda x: 1 if x > 3 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df6["rolling_mean"] = df6["System_pct"].rolling(window=10).mean()
df6["rolling_std"] = df6["System_pct"].rolling(window=10).std()

df6["anomaly_ts"] = ((df6["System_pct"] - df6["rolling_mean"]) > (2.8 * df6["rolling_std"])).astype(int)

# Combine the two methods
df6["anomaly"] = (df6["anomaly_z"] + df6["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df6["@timestamp"], df6["System_pct"])
plt.plot(df6.loc[df6["anomaly"] == 1, "@timestamp"], df6.loc[df6["anomaly"] == 1, "System_pct"], 'ro')
plt.show()
