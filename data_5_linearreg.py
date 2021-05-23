import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url="http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
x=["symboling","normalized-losses","make","fuel-type","aspiration","num of doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","no of cylinders","engine size","fuel system","bore","stroke","compression ratio","horsepower","peakrpm","city-mpg","highway-mpg","price"]
df=pd.read_csv(url)
df.columns=x
df.replace("?", np.nan, inplace = True)
a=df.head(10)
print(a)
df.dropna(subset=["price"], axis=0, inplace=True)
b=df.head(10)
print(b)
df.dropna(subset=["highway-mpg"], axis=0, inplace=True)
c=df.head(10)
print(c)
df[["price"]]=df[["price"]].astype("int")
d=df["price"]
print(d)
#Linear regression
lm=LinearRegression()
x=df[["highway-mpg"]]
y=df[["price"]]
lm.fit(x,y)
yhat=lm.predict(x)
e=yhat[0:5]
print(e)
print(lm.intercept_)
print(lm.coef_)
#insample evaluation

mse=mean_squared_error(df["price"],yhat)
print(mse)

