
import csv
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
a, b = make_blobs(n_samples=50, centers=2,random_state=0, cluster_std=0.60)

'''#(y=rank, x1= tempo, x2=energy, x3=dance, x4=loudness, x5=liveness,
x6= valence, x7=duration, x8=acousticness, x9=speechiness)'''
y=np.zeros(999)
x1=np.zeros(999)
x4=np.zeros(999)
x6=np.zeros(999)
x8=np.zeros(999)
genre=[]
with open('Top50_bare.csv', 'r') as Top50_csv:
    csv_reader=csv.reader(Top50_csv)
    size=0
    next(csv_reader)
    for line in csv_reader:
        y[size]=line[0]
        x1[size]=line[1]
        x4[size]=line[4]
        x6[size]=line[6]
        x8[size]=line[8]        
        if line[10]=='christmas':
            genre.append('r')
        else:
            genre.append('b')
        size+=1

X1=x1[:,np.newaxis]
X4=x4[:,np.newaxis]
X6=x6[:,np.newaxis]
X8=x8[:,np.newaxis]

#fig,axs=plt.subplots(2,1)

temp_val=np.concatenate((X1,X8),axis=1)

y=np.array(genre)
X1_train, X1_test, y_train, y_test = train_test_split(temp_val, y, test_size=0.2, random_state=0)
clf=svm.SVC()
clf.fit(X1_train,y_train)

def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    

plt.scatter(X1_train[:,0],X1_train[:,1],c=y_train,s=1,cmap='spring')
plot_svc_decision_function(clf);
#clf.predict(X1_test)
#axs[0].scatter(X1_train[:0],X1_train[:1],c=genre,label=genre,s=2)
#axs[0].set_title('tempo/acousticness')

#axs[1].scatter(X4,X6,c=genre,label=genre,s=2)
#axs[1].set_title('loudness/valence')
