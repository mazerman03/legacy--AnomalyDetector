import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the clean data 
df5 = pd.read_csv("cpu_QA.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df5["IDLE_pct"] = 1 - (df5["Idle"]/df5["Core"])
df5["z_score"] = stats.zscore(df5["IDLE_pct"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df5["anomaly_z"] = df5["z_score"].apply(lambda x: 1 if x > 2.8 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df5["rolling_mean"] = df5["IDLE_pct"].rolling(window=10).mean()
df5["rolling_std"] = df5["IDLE_pct"].rolling(window=10).std()

df5["anomaly_ts"] = ((df5["IDLE_pct"] - df5["rolling_mean"]) > (2.7 * df5["rolling_std"])).astype(int)

# Combine the two methods
df5["anomaly"] = (df5["anomaly_z"] + df5["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df5["@timestamp"], df5["IDLE_pct"])
plt.plot(df5.loc[df5["anomaly"] == 1, "@timestamp"], df5.loc[df5["anomaly"] == 1, "IDLE_pct"], 'ro')
plt.show()
