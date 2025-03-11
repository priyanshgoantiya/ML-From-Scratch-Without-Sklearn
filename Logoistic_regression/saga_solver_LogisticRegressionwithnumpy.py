import numpy as np
class saga_solver_LogisticRegressionwithnumpy:
  def __init__(self,learning_rate=0.01,epochs=1000,penalty=None,C=1.0,l1_ratio=0.5):
    self.lr=learning_rate
    self.iteration=epochs
    self.penalty=penalty
    self.C=C
    self.l1_ratio=l1_ratio
    self.coef_=None
    self.intercept_=None
  def sigmoid(self,z):
    z = np.clip(z, -500, 500)
    return (1/(1+np.exp(-z)))
  def fit(self,X_train,y_train):
    X_train=np.insert(X_train,0,1,axis=1)
    weights=np.zeros(X_train.shape[1])
    n_samples,n_features=X_train.shape
    gradient_memmory=np.zeros((n_samples,n_features))
    avg_gradient=np.zeros(n_features)
    for _ in range(self.iteration):
      i=np.random.randint(0,n_samples)
      y_hat=self.sigmoid(np.dot(X_train[i],weights))
      gradient=X_train[i]*(y_train[i]-y_hat)
      if self.penalty=='l2':
        gradient+=(1/self.C)* np.clip(weights, -10, 10)
      elif self.penalty=='l1':
        gradient+=(1/self.C)*np.sign(weights)
      elif self.penalty=='elasticnet':
        l1_term=self.l1_ratio*np.sign(weights)
        l2_term=(1-self.l1_ratio)*weights
        gradient+=(1/self.C)*np.sign(l1_term+l2_term)
      avg_gradient += (gradient - gradient_memmory[i]) / n_samples
      gradient_memmory[i] = gradient
      weights -= self.lr * avg_gradient
    self.coef_=weights[1:]
    self.intercept_=weights[0]
    return self.coef_,self.intercept_
  def predict_prob(self,X_test):
    X_test=np.insert(X_test,0,1,axis=1)
    return self.sigmoid(np.dot(X_test,np.concatenate(([self.intercept_],self.coef_))))
  def predict(self,X_test,threshold=0.5):
    return (self.predict_prob(X_test)>=threshold).astype(int)
