# Ridge_regression_for_ndim_from_scratch
import numpy as np 
class Ridge_regression_for_ndim_from_scratch:
  def __init__(self,alpha=0.01):
    self.alpha=alpha
    self.m=None
    self.b=None
  def fit(self,X_train,y_train):
    n_samples,n_features=X_train.shape
    X_train=np.insert(X_train,0,1,axis=1)
    I=np.identity(X_train.shape[1])
    W=np.linalg.inv(np.dot(X_train.T,X_train)+ self.alpha*I).dot(X_train.T).dot(y_train)
    self.b=W[0]
    self.m=W[1:]
  def predict(self,X_test):
    return np.dot(X_test,self.m) + self.b
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
