## simple_Ridge_regression_from_scratch
import numpy as np
class Ridge_regression_from_scratch:
  def __init__(self,alpha=0.01):
    self.alpha=alpha
    self.m=None
    self.b=None
  def fit(self,X_train,y_train):
    num=0
    den=0
    for i in range(len(X_train)):
      num=num + (y_train[i]-np.mean(y_train)) * (X_train[i]-np.mean(X_train))
      den= den + (X_train[i]-np.mean(X_train)) * (X_train[i]-np.mean(X_train))
    self.m=num/(den+ self.alpha)
    self.b=np.mean(y_train)-[(self.m)*np.mean(X_train)]
  def predict(self,X_test):
    return np.array([self.m*x + self.b for x in X_test])
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
