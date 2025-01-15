import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df2 = pd.read_csv("network_f.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df2["z_score"] = stats.zscore(df2["out"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df2["anomaly_z"] = df2["z_score"].apply(lambda x: 1 if x > 2 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df2["rolling_mean"] = df2["out"].rolling(window=10).mean()
df2["rolling_std"] = df2["out"].rolling(window=10).std()

df2["anomaly_ts"] = ((df2["out"] - df2["rolling_mean"]) > (2 * df2["rolling_std"])).astype(int)

# Combine the two methods
df2["anomaly"] = (df2["anomaly_z"] + df2["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df2["@timestamp"], df2["out"])
plt.plot(df2.loc[df2["anomaly"] == 1, "@timestamp"], df2.loc[df2["anomaly"] == 1, "out"], 'ro')
plt.show()
