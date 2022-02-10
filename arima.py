import numpy as np
import pandas as pd
import os 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima import auto_arima

def rmsd(x,y):
    z=x-y
    zs=np.square(z)
    rmsd=np.sqrt(np.mean(zs))
    return rmsd
rmsdlist=[]
nifty50=os.listdir("newdata")
count=np.arange(1,51,1)
for stock in nifty50:
    file="newdata/"+str(stock)
    df=pd.read_csv(file)
    df=df["Close"]
    ts=int(0.9*len(df))
    df_train=df.iloc[0:ts,]
    df_test=df.iloc[ts:,]
    model=auto_arima(df_train)
    pred=model.predict(n_periods=len(df_test))
    rmsdlist.append(rmsd(df_test.values,pred))
    print(rmsd(df_test.values,pred))
    
rm=np.asarray(rmsdlist)
aver=np.mean(rm)
print(aver)
#average error of 447
plt.bar(count,rmsdlist)
plt.axhline(y=aver,color='red')
plt.show()

