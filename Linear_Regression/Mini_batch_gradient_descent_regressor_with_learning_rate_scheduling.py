# Mini_batch_gradient_descent_regressor_with_learning_rate_scheduling
import numpy
class Mini_batch_gradient_descent_regressor_with_learning_rate_scheduling:
  def __init__(self,batch_size,learning_rate=0.01,epochs=100):
    self.coef_ = None
    self.intercept_ = None
    self.lr=learning_rate
    self.iteration=epochs
    self.batch_size=batch_size
    self.t0=5
    self.t1=50
  def learning_schedule(self,t):
    return self.t0/np.sqrt(self.t1+t)
  def fit(self,X_train,y_train):
    n_samples, n_features = X_train.shape
    self.coef_=np.random.randn(n_features)
    self.intercept_=np.random.randn()
    n_batches=max(n_samples//self.batch_size,1)
    for i in range(self.iteration):
      for j in range(n_batches):
        indices=np.random.choice(n_samples,self.batch_size,replace=True)
        X_batch=X_train[indices]
        y_batch=y_train[indices]
        lr=self.learning_schedule(i*(n_batches)+j)
        # update all coef and the intercept
        y_hat=np.dot(X_batch,self.coef_) + self.intercept_
        slope_intercept=(-2) * np.mean(y_batch-y_hat)
        self.intercept_=self.intercept_ - (lr * slope_intercept)
        slope_coef=(-2/self.batch_size) * np.dot((y_batch-y_hat),X_batch)
        self.coef_=self.coef_ - (lr * slope_coef)
  def predict(self,X_test):
    y_pred=np.dot(X_test,self.coef_) + self.intercept_
    return y_pred
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
