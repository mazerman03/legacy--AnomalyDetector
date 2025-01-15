import pandas as pd
import plotly.express as px


df = pd.read_csv('cpu_DEVELOP.csv',  parse_dates=["@timestamp"])



df['IDLE_pct'] = df['Idle']/df['Core']
fig = px.line(df, x='@timestamp', y='IDLE_pct', title='CPU Idle percentage')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(step="all")
        ])
    )
)
fig.show()

df.set_index('@timestamp', inplace=True)

ts1 = df['@timestamp'].values()


from statsmodels.tsa.stattools import adfuller

# assuming your time series is stored in a variable called 'ts'
result1 = adfuller(ts1)

# print the ADF test statistic and p-value
print('ADF Statistic:', result1[0])
print('p-value:', result1[1])


