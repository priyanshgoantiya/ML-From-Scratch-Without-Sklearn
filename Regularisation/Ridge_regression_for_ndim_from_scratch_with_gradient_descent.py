import numpy as np
class Ridge_regression_for_ndim_from_scratch_with_gradient_descent:
  def __init__(self,alpha=0.01,learning_rate=0.01,epochs=100):
    self.alpha=alpha
    self.learning_rate=learning_rate
    self.epochs=epochs
    self.coef_=None
    self.intercept_=None
  def fit(self,X_train,y_train):
    n_samples,n_features=X_train.shape
    self.coef_=np.zeros(n_features)
    self.intercept_=0
    parameter=np.insert(self.coef_,0,self.intercept_)
    X_train=np.insert(X_train,0,1,axis=1)
    for i in range(self.epochs):
      parameter_derivative=(1/n_samples) * (np.dot(X_train.T,X_train).dot(parameter)- np.dot(X_train.T,y_train) + self.alpha*parameter)
      parameter=parameter -(self.learning_rate * parameter_derivative)
    self.coef_=parameter[1:]
    self.intercept_=parameter[0]
  def predict(self,X_test):
    return np.dot(X_test,self.coef_) + self.intercept_
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
