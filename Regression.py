import csv
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

'''#(y=rank, x1= tempo, x2=energy, x3=dance, x4=loudness, x5=liveness,
x6= valence, x7=duration, x8=acousticness, x9=speechiness)'''
y=np.zeros(999)
x1=np.zeros(999)
x2=np.zeros(999)
x3=np.zeros(999)
x4=np.zeros(999)
x5=np.zeros(999)
x6=np.zeros(999)
x7=np.zeros(999)
x8=np.zeros(999)
x9=np.zeros(999)
with open('Top50_bare.csv', 'r') as Top50_csv:
    csv_reader=csv.reader(Top50_csv)
    size=0
    next(csv_reader)
    for line in csv_reader:
        y[size]=line[0]
        x1[size]=line[1]
        x2[size]=line[2]
        x3[size]=line[3]
        x4[size]=line[4]
        x5[size]=line[5]
        x6[size]=line[6]
        x7[size]=line[7]
        x8[size]=line[8]
        x9[size]=line[9]
        size+=1

X1=x1[:,np.newaxis]
X2=x2[:,np.newaxis]
X3=x3[:,np.newaxis]
X4=x4[:,np.newaxis]
X5=x5[:,np.newaxis]
X6=x6[:,np.newaxis]
X7=x7[:,np.newaxis]
X8=x8[:,np.newaxis]
X9=x9[:,np.newaxis]

fig,axs=plt.subplots(3,3)

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X1_train,y_train)
y_test=model.predict(X1_test)
axs[0,0].scatter(X1_train.ravel(), y_train,s=0.1)
axs[0,0].plot(X1_test.ravel(), y_test)
axs[0,0].set_title('rank/tempo')
print(model.coef_)
print(model.intercept_)

X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X2_train,y_train)
y_test=model.predict(X2_test)
axs[0,1].scatter(X2_train.ravel(), y_train,c='b',s=0.1)
axs[0,1].plot(X2_test.ravel(), y_test,c='b')
axs[0,1].set_title('rank/energy')
print(model.coef_)
print(model.intercept_)

X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X3_train,y_train)
y_test=model.predict(X3_test)
axs[0,2].scatter(X3_train.ravel(), y_train,c='g',s=0.1)
axs[0,2].plot(X3_test.ravel(), y_test,)
axs[0,2].set_title('rank/dance')
print(model.coef_)
print(model.intercept_)

X4_train, X4_test, y_train, y_test = train_test_split(X4, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X4_train,y_train)
y_test=model.predict(X4_test)
axs[1,0].scatter(X4_train.ravel(), y_train,c='r',s=0.1)
axs[1,0].plot(X4_test.ravel(), y_test)
axs[1,0].set_title('rank/loudness')
print(model.coef_)
print(model.intercept_)

X5_train, X5_test, y_train, y_test = train_test_split(X5, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X5_train,y_train)
y_test=model.predict(X5_test)
axs[1,1].scatter(X5_train.ravel(), y_train,c='c',s=0.1)
axs[1,1].plot(X5_test.ravel(), y_test)
axs[1,1].set_title('rank/liveness')
print(model.coef_)
print(model.intercept_)

X6_train, X6_test, y_train, y_test = train_test_split(X6, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X6_train,y_train)
y_test=model.predict(X6_test)
axs[1,2].scatter(X6_train.ravel(), y_train,c='m',s=0.1)
axs[1,2].plot(X6_test.ravel(), y_test)
axs[1,2].set_title('rank/valence')
print(model.coef_)
print(model.intercept_)

X7_train, X7_test, y_train, y_test = train_test_split(X7, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X7_train,y_train)
y_test=model.predict(X7_test)
axs[2,0].scatter(X7_train.ravel(), y_train,c='y',s=0.1)
axs[2,0].plot(X7_test.ravel(), y_test)
axs[2,0].set_title('rank/duration')
print(model.coef_)
print(model.intercept_)

X8_train, X8_test, y_train, y_test = train_test_split(X8, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X8_train,y_train)
y_test=model.predict(X8_test)
axs[2,1].scatter(X8_train.ravel(), y_train,c='k',s=0.1)
axs[2,1].plot(X8_test.ravel(), y_test)
axs[2,1].set_title('rank/acousticness')
print(model.coef_)
print(model.intercept_)

X9_train, X9_test, y_train, y_test = train_test_split(X9, y, test_size=0.2, random_state=0)
model=LinearRegression(normalize=True)
model.fit(X9_train,y_train)
y_test=model.predict(X9_test)
axs[2,2].scatter(X9_train.ravel(), y_train,c='tab:orange',s=0.1)
axs[2,2].plot(X9_test.ravel(), y_test)
axs[2,2].set_title('rank/speechiness')
print(model.coef_)
print(model.intercept_)