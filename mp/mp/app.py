from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def posix_time(dt):
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)


data = pd.read_csv(' mp/traffic_volume_data.csv')
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
data.to_csv('mp//traffic_volume_data.csv', index=None)

sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv("traffic_volume_data.csv")
data = data.sample(700).reset_index(drop=True)
label_columns = ['source', 'destination','route']
numeric_columns = ['is_holiday', 'weekday', 'hour', 'day', 'year', 'month']

features = numeric_columns + label_columns
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

new = []
print(regr.predict(X[:10]))
print(y[:10])
#####################
app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('index.html')


d = {}


@app.route('/predict', methods=['POST'])
def predict():
    d['route'] = request.form.get('x2')
    if d['route'] == 'lohgaon road':
        d['route'] = int(1)
    else:
        d['route'] = int(0)
    d['is_holiday'] = request.form['isholiday']
    if d['is_holiday'] == 'yes':
        d['is_holiday'] = int(1)
    else:
        d['is_holiday'] = int(0)
   
    d['weekday'] = int(0)
    D = request.form['date']
    d['hour'] = int(request.form['time'][:2])
    d['day'] = int(D[8:])
    d['year'] = int(D[:4])
 
    d['month'] = int(D[5:7])
    d['x0'] = request.form.get('x0')
   
    d['x1'] = request.form.get('x1')
    
    xoval = {'x0_nigadi', 'x0_wagholi'}
    x1val = {'x1_wagholi',
             'x1_nigadi'}
    x2val = {'x2_lohagaon road','x2_khadaki'}
    
    x0 = {}
    x1 = {}
    
    for i in xoval:
        x0[i] = 0
    for i in x1val:
        x1[i] = 0
     
    x0[d['x0']] = 1
    x1[d['x1']] = 1
   
   
    final = []
    final.append(d['is_holiday'])
    final.append(d['route'])
    final.append(d['weekday'])
    final.append(d['hour'])
    final.append(d['day'])
    final.append(d['year'])
    final.append(d['month'])
    for i in x0:
        final.append(x0[i])
    for i in x1:
        final.append(x1[i])
    
    output = print(regr.predict([final]))
    print(output)
    return render_template('output.html', data1=d, data2=final)
    
   


if __name__ == '__main__':
    app.run(debug=True)
