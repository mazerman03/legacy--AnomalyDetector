import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df7 = pd.read_csv("cpu_DEVELOP.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df7["User_pct"] = (df7["User"]/df7["Core"])
df7["z_score"] = stats.zscore(df7["User_pct"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df7["anomaly_z"] = df7["z_score"].apply(lambda x: 1 if x > 2.75 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df7["rolling_mean"] = df7["User_pct"].rolling(window=10).mean()
df7["rolling_std"] = df7["User_pct"].rolling(window=10).std()

df7["anomaly_ts"] = ((df7["User_pct"] - df7["rolling_mean"]) > (2.75 * df7["rolling_std"])).astype(int)

# Combine the two methods
df7["anomaly"] = (df7["anomaly_z"] + df7["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df7["@timestamp"], df7["User_pct"])
plt.plot(df7.loc[df7["anomaly"] == 1, "@timestamp"], df7.loc[df7["anomaly"] == 1, "User_pct"], 'ro')
plt.show()
