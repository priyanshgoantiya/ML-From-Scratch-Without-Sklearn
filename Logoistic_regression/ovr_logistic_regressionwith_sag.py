import numpy as np
class ovr_logistic_regressionwith_sag:
  def __init__(self,learning_rate=0.01,epochs=100):
    self.lr=learning_rate
    self.iteration=epochs
    self.weights=None
    self.bias=None
    self.classes=None
  def sigmoid(self,z):
    z = np.clip(z, -100, 100)
    return (1/(1+np.exp(-z)))
  def train_binary(self,X_train,y_train):
    n_samples,n_features=X_train.shape
    weights=np.zeros(n_features)
    bias=0
    for _ in range(self.iteration):
      for i in range(n_samples):
        indices=np.random.randint(0,n_samples)
        y_hat=self.sigmoid(np.dot(X_train[indices],weights)+bias)
        derivative_weights=X_train[indices] *(y_hat-y_train[indices])
        derivative_bias=np.sum(y_hat-y_train[indices])
        weights=weights - (self.lr*(derivative_weights))
        bias=bias - (self.lr*derivative_bias)
    return weights,bias
  def fit(self,X_train,y_train):
    self.weights={}
    self.bias={}
    self.classes=np.unique(y_train)
    for clas in self.classes:
      y_binary=np.where(y_train==clas,1,0)
      self.weights[clas],self.bias[clas]=self.train_binary(X_train,y_binary)
  def predict_prob(self,X_test):
    probs=np.array([self.sigmoid((np.dot(X_test,self.weights[clas]))+ self.bias[clas]) for clas in self.classes])
    return probs.T
  def predict(self,X_test):
    probs=self.predict_prob(X_test)
    return self.classes[np.argmax(probs,axis=1)]
