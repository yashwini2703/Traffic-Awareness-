from datetime import datetime
from functools import reduce
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error



def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)


n1features = []
n2features = []
n3features = []
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)

app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/train')
def train():
    data = pd.read_csv('traffic_volume_data.csv')
    data = data.sort_values(
        by=['date_time'], ascending=True).reset_index(drop=True)

    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)

    data['date_time'] = pd.to_datetime(data['date_time'])

    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
    data.to_csv("traffic_volume_data.csv", index=None)

    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')
    data = pd.read_csv("traffic_volume_data.csv")
    data = data.sample(700).reset_index(drop=True)
    label_columns = ['source', 'destination','route']
    numeric_columns = ['is_holiday', 'weekday', 'hour', 'day', 'year', 'month']
    n1 = data['source']
    n2 = data['destination']
    n3 = data['route']
    unique(n1)
    unique(n2)
    unique(n3)
    n1features = ['nigadi', 'wagholi']
    n2features = [ 'nigadi', 'wagholi']
    n3features = ['lohgaon road', 'khadaki']
    """#Data Preparation"""
    n11 = []
    n22 = []
    n33 = []
    for i in range(700):
        if(n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1
                                         [i]))+1)
        if n2[i] not in n2features:
            n22.append(0)
        else:
            n22.append((n2features.index(n2[i]))+1)
        if n3[i] not in n3features:
            n33.append(0)
        else:
            n33.append((n3features.index(n3[i]))+1)    

  
    data['source'] = n11
    data['destination'] = n22
    data['route'] = n33

    features = numeric_columns+label_columns
    target = ['traffic']
    X = data[features]
    y = data[target]
    print(X)
    print(data[features].hist(bins=20,))

    data['traffic'].hist(bins=20)

    """#Feature Scaling"""

    X = x_scaler.fit_transform(X)

    y = y_scaler.fit_transform(y).flatten()
    print(X)
    warnings.filterwarnings('ignore')

    regr.fit(X, y)
    # error eval
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
    ##############################
    print('predicted output :=', regr.predict(X[:10]))
    print('Actual output :=', y[:10])
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ip = []
    if request.form['isholiday'] == 'yes':
        ip.append(1)
    else:
        ip.append(0)

    selected_date = request.form['date']

    if not isinstance(selected_date, datetime):
        selected_date = datetime.strptime(selected_date, "%Y-%m-%d")

    weekday = selected_date.weekday() + 1  
    ip.append(weekday)

    ip.append(int(request.form['time'][:2]))
    D = request.form['date']
    ip.append(int(D[8:]))
    ip.append(int(D[:4]))
    ip.append(int(D[5:7]))

    s1 = request.form.get('x0')
    s2 = request.form.get('x1')
    s3 = request.form.get('x2')
    if(s1) not in n1features:
        ip.append(0)
    else:
        ip.append((n1features.index(s1))+1)
    if s2 not in n2features:
        ip.append(0)
    else:
        ip.append((n2features.index(s2))+1)
    if s3 not in n3features:
        ip.append(0)
    else:
        ip.append((n3features.index(s3))+1)    
    ip = x_scaler.transform([ip])
    out = regr.predict(ip)
    print('Before inverse Scaling :', out)
    y_pred = y_scaler.inverse_transform([out])
    print('Traffic Volume : ', y_pred)
    s = ''
    if(y_pred <= 750):
        print("Minimal Traffic ")
        s = "Minimal Traffic "
    elif y_pred > 750 and y_pred <= 1500:
        print("Very Light Traffic")
        s = "Very Light Traffic"
    elif y_pred > 1500 and y_pred <= 2250:
        print("Light Traffic")
        s = "Light Traffic"
    elif y_pred > 2250 and y_pred <= 3000:
        print("Light to Moderate Traffic")
        s = "Light to Moderate Traffic"
    elif y_pred > 3000 and y_pred <= 3750:
        print("Moderate Traffic")
        s = "Moderate Traffic"
    elif y_pred > 3750 and y_pred <= 4500:
        print("Moderate To Heavy Traffic")
        s = "Moderate to Heavy Traffic"               
    elif y_pred > 4500 and y_pred <= 5250:
        print("Heavy Traffic")
        s = "Heavy Traffic"
    elif y_pred > 4500 and y_pred <= 5250:
        print("Very Heavy Traffic")
        s = "Very Heavy Traffic"
    else:
        print("Worst case")
        s = "Worst case"
    return render_template('output.html', data1=ip, op=y_pred, statement=s)


if __name__ == '__main__':
    app.run(debug=True)
