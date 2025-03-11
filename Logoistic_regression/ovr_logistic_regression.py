# ovr_logistic_regression_for_multiclass_classification
import numpy as np 
class ovr_logistic_regression:
  def __init__(self,learning_rate=0.01,epochs=1000):
    self.lr=learning_rate
    self.iteration=epochs
    self.weights={}
    self.bias={}
  def sigmoid(self,z):
    z = np.clip(z, -500, 500)
    return (1/(1+np.exp(-z)))
  def train_binary(self,X_train,y_train):
    n_samples,n_features=X_train.shape
    weights=np.ones(n_features)
    bias=0
    for _ in range(self.iteration):
      y_hat=self.sigmoid(np.dot(X_train,weights)+bias)
      derivative_weights=(1/n_samples)*np.dot(X_train.T,(y_hat-y_train))
      derivative_bias=(1/n_samples)*np.sum(y_hat-y_train)
      weights=weights - (self.lr*(derivative_weights))
      bias=bias - (self.lr*derivative_bias)
    return weights,bias
  def fit(self,X_train,y_train):
    self.classes=np.unique(y_train)
    for clas in self.classes:
      y_binary=np.where(y_train==clas,1,0)
      self.weights[clas],self.bias[clas]=self.train_binary(X_train,y_binary)
  def predict_prob(self,X_test):
    probs={}
    for clas in self.classes:
      probs[clas]=self.sigmoid((np.dot(X_test,self.weights[clas]))+ self.bias[clas])
    return probs
  def predict(self,X_test):
    probs=self.predict_prob(X_test)
    return np.array([max(probs ,key=lambda clas: probs[clas][i]) for i in range(X_test.shape[0])])
