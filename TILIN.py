import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df1= pd.read_csv("network_f.csv", parse_dates=["@timestamp"])
df2 = pd.read_csv("network_f.csv", parse_dates=["@timestamp"])
df3 = pd.read_csv("memory_DEVELOP.csv", parse_dates=["@timestamp"])
df4 = pd.read_csv("memory_DEVELOP.csv", parse_dates=["@timestamp"])
df5 = pd.read_csv("cpu_DEVELOP.csv", parse_dates=["@timestamp"])
df6 = pd.read_csv("cpu_DEVELOP.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df1["z_score"] = stats.zscore(df1["in"])
df2["z_score"] = stats.zscore(df2["out"])
df3["z_score"] = stats.zscore(df3["Actpercen"])
df4["SWAP"] = (1 - df4["Free"]/df4["Total"])
df4["z_score"] = stats.zscore(df4["SWAP"])
df5["z_score"] = stats.zscore(df5["Idle"]/df5["Core"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df1["anomaly_z"] = df1["z_score"].apply(lambda x: 1 if x > 2 else 0)
df2["anomaly_z"] = df2["z_score"].apply(lambda x: 1 if x > 2 else 0)
df3["anomaly_z"] = df3["z_score"].apply(lambda x: 1 if x > 2 else 0)
df4["anomaly_z"] = df4["z_score"].apply(lambda x: 1 if x > 2 else 0)
df5["anomaly_z"] = df5["z_score"].apply(lambda x: 1 if x > 2.8 else 0)


#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df1["rolling_mean"] = df1["in"].rolling(window=10).mean()
df1["rolling_std"] = df1["in"].rolling(window=10).std()

df1["anomaly_ts"] = ((df1["in"] - df1["rolling_mean"]) > (3 * df1["rolling_std"])).astype(int)

df2["rolling_mean"] = df2["out"].rolling(window=10).mean()
df2["rolling_std"] = df2["out"].rolling(window=10).std()

df2["anomaly_ts"] = ((df2["out"] - df2["rolling_mean"]) > (3 * df2["rolling_std"])).astype(int)

df3["rolling_mean"] = df3["Actpercen"].rolling(window=10).mean()
df3["rolling_std"] = df3["Actpercen"].rolling(window=10).std()

df3["anomaly_ts"] = ((df3["Actpercen"] - df3["rolling_mean"]) > (3 * df3["rolling_std"])).astype(int)

df4["rolling_mean"] = df4["SWAP"].rolling(window=10).mean()
df4["rolling_std"] = df4["SWAP"].rolling(window=10).std()

df4["anomaly_ts"] = ((df4["SWAP"] - df4["rolling_mean"]) > (3 * df4["rolling_std"])).astype(int)

df5["rolling_mean"] = df5["IDLE_pct"].rolling(window=10).mean()
df5["rolling_std"] = df5["IDLE_pct"].rolling(window=10).std()

df5["anomaly_ts"] = ((df5["IDLE_pct"] - df5["rolling_mean"]) > (2.7 * df5["rolling_std"])).astype(int)


# Combine the two methods
df1["anomaly"] = (df1["anomaly_z"] + df1["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df1["@timestamp"], df1["in"])
plt.plot(df1.loc[df1["anomaly"] == 1, "@timestamp"], df1.loc[df1["anomaly"] == 1, "in"], 'ro')

df2["anomaly"] = (df2["anomaly_z"] + df2["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df2["@timestamp"], df2["out"])
plt.plot(df2.loc[df2["anomaly"] == 1, "@timestamp"], df2.loc[df2["anomaly"] == 1, "out"], 'ro')


df3["anomaly"] = (df3["anomaly_z"] + df3["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df3["@timestamp"], df3["Actpercen"])
plt.plot(df3.loc[df3["anomaly"] == 1, "@timestamp"], df3.loc[df3["anomaly"] == 1, "Actpercen"], 'ro')


df4["anomaly"] = (df4["anomaly_z"] + df4["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df4["@timestamp"], df4["SWAP"])
plt.plot(df4.loc[df4["anomaly"] == 1, "@timestamp"], df4.loc[df4["anomaly"] == 1, "SWAP"], 'ro')

df5["anomaly"] = (df5["anomaly_z"] + df5["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df5["@timestamp"], df5["IDLE_pct"])
plt.plot(df5.loc[df5["anomaly"] == 1, "@timestamp"], df5.loc[df5["anomaly"] == 1, "IDLE_pct"], 'ro')
plt.show()



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