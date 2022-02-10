import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
import seaborn as sns

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
    data=pd.read_csv(file)
    data=data.drop(data.columns[[1,2,3,4,5,6,7,9,10,11,12,13,14]], axis=1)

    ind=[]
    data_train=data.iloc[0:1000,]
    for i in range(len(data_train)):
        ind.append(i)
    ind=np.asarray(ind)
    ind=ind.reshape(-1,1)

    poly=PF(5)
    x=poly.fit_transform(ind)
    y=np.asarray(data_train['Close'])
    y=y.reshape(-1,1)
    model=LR()
    model.fit(x,y)
    ypred=model.predict(x)


    data_test=data.iloc[1000:,]
    ind=[]
    for i in range(len(data_test)):
        ind.append(i)
    ind=np.asarray(ind)
    ind=ind.reshape(-1,1)
    xtest=poly.fit_transform(ind)
    ytest=model.predict(xtest)
    yhat=np.asarray(data_test['Close'])

    error=rmsd(yhat,ytest)
    rmsdlist.append(error)
    print(error)
    
rm=np.asarray(rmsdlist)
aver=np.mean(rm)
print(aver)
#average error of 1250
plt.bar(count,rmsdlist)
plt.axhline(y=aver,color='red')
plt.show()