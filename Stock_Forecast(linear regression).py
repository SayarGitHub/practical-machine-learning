import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

df = quandl.get("WIKI/GOOGL")
df.to_csv("Google_Stocks.csv")
df = pd.read_csv("Google_Stocks.csv", index_col="Date")

col = df.columns

df.drop(columns=col[0:7], inplace= True)

df["HL_PCT"]=(df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"]*100.0

df["PCT_change"]=(df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"]*100.0

df.drop(columns=["Adj. Open","Adj. High","Adj. Low"],inplace=True)

forecast_col = "Adj. Close"
df.fillna(-999999,inplace=True) #replace NaN with outliers

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df["label"] = df[forecast_col].shift(-forecast_out) #label column is Adj. Close shifted up by 35 days

X=np.array(df.drop(columns='label'))       #no.of examples*no.of features
X = preprocessing.scale(X)                 #feature scaling
X_predict = X[-forecast_out:]              #these feature values will be used to predict 35 days in future
print(X_predict.shape)
X = X[:-forecast_out]                      #for training and testing
print(X.shape)

df.dropna(inplace=True) #drops the last 35 rows from df
y=np.array(df["label"]) #no.of examples*1
print(y.shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2) #shuffle and split

clf = LinearRegression(n_jobs=-1) #n_jobs=-1 =>use all processing power for other values it represents parallel threads

clf.fit(X_train,y_train)
with open("linearregression.pickle","wb") as f:
    pickle.dump(clf,f)
pickle_in = open("linearregression.pickle","rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
print(accuracy) 

forecast_set = clf.predict(X_predict)
print(forecast_set)

df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix+one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc = 4) #legend in 4th quadrant
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()