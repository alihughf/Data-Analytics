
import csv
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

'''#(y=rank, x1= tempo, x2=energy, x3=dance, x4=loudness, x5=liveness,
x6= valence, x7=duration, x8=acousticness, x9=speechiness)'''
y=np.zeros(999)
x1=np.zeros(999)
x4=np.zeros(999)
x6=np.zeros(999)
x8=np.zeros(999)
genre=[]
cl=[]
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
            cl.append(0)
        else:
            genre.append('b')
            cl.append(1)
        size+=1

X1=x1[:,np.newaxis]
X4=x4[:,np.newaxis]
X6=x6[:,np.newaxis]
X8=x8[:,np.newaxis]

#fig,axs=plt.subplots(2,1)
clf=svm.SVC()

temp_acc=np.concatenate((X1,X8),axis=1)
loud_val=np.concatenate((X4,X6),axis=1)
y=np.array(cl)

X1_train, X1_test, y1_train, y1_test = train_test_split(temp_acc, y, test_size=0.2, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(loud_val, y, test_size=0.2, random_state=0)
genre1_train=[]
genre2_train=[]
for i in y1_train:
    if i ==0:
        genre1_train.append('r')
    if i==1:
        genre1_train.append('b')
for i in y2_train:
    if i ==0:
        genre2_train.append('r')
    if i==1:
        genre2_train.append('b')        

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

clf.fit(X1_train,y1_train)


fig,axs=plt.subplots(1,2)
xx1, yy1 =make_meshgrid(X1_train[:,0],X1_train[:,1])
xx2,yy2=make_meshgrid(X2_train[:,0],X2_train[:,1])

Z1=clf.predict(np.c_[xx1.ravel(),yy1.ravel()])
Z1=Z1.reshape(xx1.shape)
pred1=clf.predict(np.c_[X1_test[:,0].ravel(),X1_test[:,1].ravel()])
print(confusion_matrix(y1_test,pred1))
axs[0].contourf(xx1,yy1,Z1)

axs[0].scatter(X1_train[:,0],X1_train[:,1],c=genre1_train,label=genre1_train,s=2)
axs[0].set_xlim(xx1.min(), xx1.max())
axs[0].set_ylim(yy1.min(), yy1.max())
axs[0].set_xlabel('Tempo')
axs[0].set_ylabel('Acousticness')
axs[0].set_xticks(())
axs[0].set_yticks(())
axs[0].set_title('Tempo/Acousticness')


clf.fit(X2_train,y2_train)
pred2=clf.predict(np.c_[X2_test[:,0].ravel(),X2_test[:,1].ravel()])
print(confusion_matrix(y2_test,pred2))

Z2=clf.predict(np.c_[xx2.ravel(),yy2.ravel()])
Z2=Z2.reshape(xx2.shape)

axs[1].contourf(xx2,yy2,Z2)
axs[1].scatter(X2_train[:,0],X2_train[:,1],c=genre2_train,label=genre2_train,s=2)
axs[1].set_xlim(xx2.min(), xx2.max())
axs[1].set_ylim(yy2.min(), yy2.max())
axs[1].set_xlabel('Loudness')
axs[1].set_ylabel('Valence')
axs[1].set_xticks(())
axs[1].set_yticks(())
axs[1].set_title('Loudness/Valence')

'''
axs[0].scatter(X1_train[:,0],X1_train[:,1],c=y1_train,label=y1_train,s=2)
axs[0].set_title('tempo/acousticness')

axs[1].scatter(X2_train[:,0],X2_train[:,1],c=y2_train,label=y2_train,s=2)
#axs[1].set_title('loudness/valence')'''
