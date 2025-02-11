# Mini_batch_gradient_descent_regressor
import numpy 
class Mini_batch_gradient_descent_regressor:
  def __init__(self,learning_rate=0.01,epochs=100,batch_size=10):
    self.coef_ = None
    self.intercept_ = None
    self.lr=learning_rate
    self.iteration=epochs
    self.batch_size=batch_size
  def fit(self,X_train,y_train):
    n_samples, n_features = X_train.shape
    self.coef_=np.random.randn(n_features)
    self.intercept_=np.random.randn()
    n_batches=max(n_samples//self.batch_size,1)
    for i in range(self.iteration):
      for j in range(n_batches):
        index=np.random.randint(n_samples)
        # update all coef and the intercept
        y_hat=np.dot(X_train[index],self.coef_) + self.intercept_
        slope_intercept=(-2) * (y_train[index]-y_hat)
        self.intercept_=self.intercept_ - (self.lr * slope_intercept)
        slope_coef=(-2) * np.dot((y_train[index]-y_hat),X_train[index])
        self.coef_=self.coef_ - (self.lr * slope_coef)
  def predict(self,X_test):
    y_pred=np.dot(X_test,self.coef_) + self.intercept_
    return y_pred
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
