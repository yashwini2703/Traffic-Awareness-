import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('Traffic_Prediction-main/mp/traffic_volume_data.csv')
data = data.sort_values(
	by=['date_time'], ascending=True).reset_index(drop=True)

data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

data['date_time'] = pd.to_datetime(data['date_time'], format="%d/%m/%Y %H:%M:%S")

data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
data['day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
data.to_csv("traffic_volume_data.csv", index=None)
data.columns

sns.set()  
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv("traffic_volume_data.csv")
data = data.sample(700).reset_index(drop=True)
label_columns = ['source', 'destination']
numeric_columns = ['is_holiday', 'weekday', 'hour', 'day', 'year', 'month']
from sklearn.preprocessing import OneHotEncoder
ohe_encoder = OneHotEncoder()
x_ohehot = ohe_encoder.fit_transform(data[label_columns])
ohe_features = ohe_encoder.get_feature_names_out()
x_ohehot = pd.DataFrame(x_ohehot.toarray(),
						columns=ohe_features)
data = pd.concat([data[['date_time']],data[['traffic']+numeric_columns],x_ohehot],axis=1)
data['traffic'].hist(bins=20)
metrics = ['month', 'day', 'weekday', 'hour']

fig = plt.figure(figsize=(8, 4*len(metrics)))
for i, metric in enumerate(metrics):
	ax = fig.add_subplot(len(metrics), 1, i+1)
	ax.plot(data.groupby(metric)['traffic'].mean(), '-o')
	ax.set_xlabel(metric)
	ax.set_ylabel("Mean Traffic")
	ax.set_title(f"Traffic Trend by {metric}")
plt.tight_layout()
plt.show()

features = numeric_columns+list(ohe_features)
target = ['traffic']
X = data[features]
y = data[target]
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()

warnings.filterwarnings('ignore')
##################
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
print(regr.predict(X[:10]))
print(y[:10])
##################
