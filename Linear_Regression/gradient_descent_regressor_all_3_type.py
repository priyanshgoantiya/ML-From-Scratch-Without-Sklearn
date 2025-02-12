## gradient_descent_regressor_all_3_type
import numpy as np
class gradient_descent_regressor:
  def __init__(self,learning_rate=0.01,epochs=100,batch_size=10,method='Batch'):
    self.lr=learning_rate
    self.epochs=epochs
    self.batch_size=batch_size
    self.method=method
    self.t0=100
    self.t1=10
    self.coef_ = None
    self.intercept_ = None
  def learning_rate_scheduling(self, t):
    return min(self.t0 / np.sqrt(t + self.t1), 0.1)  # Cap learning rate at 0.1
  def fit(self,X_train,y_train):
    n_samples,n_features=X_train.shape
    self.coef_=np.zeros(n_features)
    self.intercept_=0
    n_batches=max(n_samples//self.batch_size,1)
    for i in range(self.epochs):
      if self.method =='Batch':
        lr=self.learning_rate_scheduling(i+1)
        y_hat=np.dot(X_train,self.coef_ ) + self.intercept_
        gradient_intercept=(-2/n_samples) * np.sum(y_train-y_hat)
        self.intercept_= self.intercept_ - (lr *  gradient_intercept)
        gradient_coef=(-2/n_samples) * np.dot((y_train-y_hat),X_train)
        self.coef_= self.coef_ - (lr *  gradient_coef)
      elif self.method=='Stochastic':
        for j in range(n_samples):
          lr=self.learning_rate_scheduling(i*n_samples+j+1)
          indices=np.random.randint(0,n_samples,size=1)
          X_sample=X_train[indices]
          y_sample=y_train[indices]
          y_hat=np.dot(X_sample,self.coef_) + self.intercept_
          gradient_intercept=(-2) * (y_sample-y_hat)
          self.intercept_= self.intercept_ - (lr *  gradient_intercept)
          gradient_coef=(-2) * np.dot((y_sample-y_hat),X_sample)          
          self.coef_= self.coef_ - (lr *  gradient_coef)
      elif self.method=='Mini Batch':
        for j in range(n_batches):
          lr=self.learning_rate_scheduling(i*n_batches+j+1)
          indices=np.random.choice(n_samples,self.batch_size,replace=True)
          X_batch=X_train[indices]
          y_batch=y_train[indices]
          y_hat=np.dot(X_batch,self.coef_) + self.intercept_
          gradient_intercept=(-2/self.batch_size) * np.sum(y_batch-y_hat)
          self.intercept_= self.intercept_ - (lr *  gradient_intercept)
          gradient_coef=(-2/self.batch_size) * np.dot((y_batch-y_hat),X_batch)          
          self.coef_= self.coef_ - (lr *  gradient_coef)
  def predict(self,X_test):
    y_pred=np.dot(X_test,self.coef_ ) + self.intercept_
    return y_pred
  
  def r2_score(self, y_test, y_pred):

    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2) + 1e-8
    return 1 - (ssr / ssm)
