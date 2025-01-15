import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df4 = pd.read_csv("memory_DEVELOP.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df4["SWAP"] = (1 - df4["Free"]/df4["Total"])
df4["z_score"] = stats.zscore(df4["SWAP"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df4["anomaly_z"] = df4["z_score"].apply(lambda x: 1 if x > 2 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df4["rolling_mean"] = df4["SWAP"].rolling(window=10).mean()
df4["rolling_std"] = df4["SWAP"].rolling(window=10).std()

df4["anomaly_ts"] = ((df4["SWAP"] - df4["rolling_mean"]) > (2 * df4["rolling_std"])).astype(int)

# Combine the two methods
df4["anomaly"] = (df4["anomaly_z"] + df4["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df4["@timestamp"], df4["SWAP"])
plt.plot(df4.loc[df4["anomaly"] == 1, "@timestamp"], df4.loc[df4["anomaly"] == 1, "SWAP"], 'ro')
plt.show()
