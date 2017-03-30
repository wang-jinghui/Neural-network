import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from load_data import *


n_estimators = 200
learning_tate = 0.69

train_x,train_y,train_q = processData('dataSet/Fold1/train.txt')
vali_x,vali_y,vali_q = processData('dataSet/Fold1/test.txt')

MLP = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',alpha=0.001)
MLP.fit(train_x, train_y)
MLP_err = zero_one_loss(MLP.predict(vali_x),vali_y)


train_x = np.array(train_x)
vali_x = np.array(vali_x)
#train_x = GaussianProce(train_x).astype('float32')
#vali_x = GaussianProce(vali_x).astype('float32')


'''
pca = PCA(45)
train_x = np.array(train_x).astype('float32')
vali_x = np.array(vali_x).astype('float32')

pca.fit(train_x)
train_x = pca.transform(train_x)

pca.fit(vali_x)
vali_x = pca.transform(vali_x)


dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(train_x, train_y)
p_vali_y = dt_stump.predict(vali_x)
dt_stump_err = zero_one_loss(p_vali_y, vali_y)
'''
dt = DecisionTreeClassifier(max_depth=11,min_samples_leaf=1,random_state=1,min_impurity_split=0.0001)
dt.fit(train_x, train_y)
dt_err = zero_one_loss(dt.predict(vali_x), vali_y)


'''
ada_discrete = AdaBoostClassifier(base_estimator=MLP,
                                  learning_rate= learning_tate,
                                  n_estimators=n_estimators,
                                  algorithm='SAMME')

ada_discrete.fit(train_x, train_y)
'''
ada_real = AdaBoostClassifier(base_estimator=dt,
                              learning_rate= learning_tate,
                              n_estimators = n_estimators,
                              algorithm='SAMME.R')
ada_real.fit(train_x, train_y)


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot([1, n_estimators], [MLP_err] *2, 'k-', label = 'knn auto')
ax.plot([1, n_estimators], [dt_err] *2, 'k--', label = 'Decision Tree')

'''
ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(vali_x)):
    ada_discrete_err[i] = zero_one_loss(vali_y, y_pred)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(train_x)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, train_y)
'''
ada_real_err = np.zeros((n_estimators,))
for i, py in enumerate(ada_real.staged_predict(vali_x)):
    ada_real_err[i] = zero_one_loss(vali_y, py)

ada_real_train = np.zeros((n_estimators,))
for i, py in enumerate(ada_real.staged_predict(train_x)):
    ada_real_train[i] =zero_one_loss(py, train_y)

x_axis = np.arange(n_estimators)

#ax.plot(x_axis+1, ada_discrete_err,label= 'ada_knn_auto test',c='red')
#ax.plot(x_axis+1, ada_discrete_err_train,label = 'ada_knn_auto train',c='blue')

ax.plot(x_axis+1, ada_real_err,label = 'ada_SAMME.R test',c='blue')
ax.plot(x_axis+1, ada_real_train,label='ada_SAMME.R train',c='green')

ax.set_ylim((0.0,0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error')

leg = ax.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()

