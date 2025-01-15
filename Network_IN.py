import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df = pd.read_csv("network_f.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df["z_score"] = stats.zscore(df["in"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df["anomaly_z"] = df["z_score"].apply(lambda x: 1 if x > 2 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df["rolling_mean"] = df["in"].rolling(window=10).mean()
df["rolling_std"] = df["in"].rolling(window=10).std()

df["anomaly_ts"] = ((df["in"] - df["rolling_mean"]) > (2 * df["rolling_std"])).astype(int)

# Combine the two methods
df["anomaly"] = (df["anomaly_z"] + df["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df["@timestamp"], df["in"])
plt.plot(df.loc[df["anomaly"] == 1, "@timestamp"], df.loc[df["anomaly"] == 1, "in"], 'ro')
plt.show()
