import numpy as np
class LogisticRegression_withnumpy:
  def __init__(self,learning_rate=0.01,epochs=1000):
    self.lr=learning_rate
    self.iteration=epochs
    self.coef_=None
    self.intercept_=None
  def sigmoid(self,z):
    return (1/(1+np.exp(-z)))
  def fit(self,X_train,y_train):
    X_train=np.insert(X_train,0,1,axis=1)
    weights=np.ones(X_train.shape[1])
    for i in range(self.iteration):
      y_hat=self.sigmoid(np.dot(X_train,weights))
      gradient=(1/X_train.shape[0])*(np.dot(X_train.T,(y_train-y_hat)))
      weights=weights + self.lr*(gradient)
    self.coef_=weights[1:]
    self.intercept_=weights[0]
    return self.coef_,self.intercept_
  def predict_prob(self,X_test):
    X_test=np.insert(X_test,0,1,axis=1)
    return self.sigmoid(np.dot(X_test,np.concatenate(([self.intercept_],self.coef_))))
  def predict(self,X_test,threshold=0.5):
    return (self.predict_prob(X_test)>=threshold).astype(int)
