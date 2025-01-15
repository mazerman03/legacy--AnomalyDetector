import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#Read the data once we separate it from the original csv
df3 = pd.read_csv("memory_DEVELOP.csv", parse_dates=["@timestamp"])

#Calculate the standard deviation for the specific variable
df3["z_score"] = stats.zscore(df3["Actpercen"])

#Apply anomaly if the standard deviation is higher than 3 (adding a column where we add 1 for anomaly according to z-score and 0 for no anomaly)
df3["anomaly_z"] = df3["z_score"].apply(lambda x: 1 if x > 2 else 0)

#Perform a time series analysis using the moving average and standard deviation (moving average to smooth out the curve and filter out random flunctations in data)
df3["rolling_mean"] = df3["Actpercen"].rolling(window=10).mean()
df3["rolling_std"] = df3["Actpercen"].rolling(window=10).std()

df3["anomaly_ts"] = ((df3["Actpercen"] - df3["rolling_mean"]) > (2 * df3["rolling_std"])).astype(int)

# Combine the two methods
df3["anomaly"] = (df3["anomaly_z"] + df3["anomaly_ts"]).apply(lambda x: 1 if x >= 2 else 0)

plt.plot(df3["@timestamp"], df3["Actpercen"])
plt.plot(df3.loc[df3["anomaly"] == 1, "@timestamp"], df3.loc[df3["anomaly"] == 1, "Actpercen"], 'ro')
plt.show()
