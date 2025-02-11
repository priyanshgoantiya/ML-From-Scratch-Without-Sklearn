# gradient_descent_regressor
import numpy 
class gradient_descent_regressor:
  def __init__(self,learning_rate,epochs):
    self.m = np.random.randn()  # Random initialization
    self.b = np.random.randn()  # Random initialization
    self.lr=learning_rate
    self.iteration=epochs
  def fit(self,X,y):
    ## clac the b by gd
    for i in range(self.iteration):
      X=X.ravel()
      loss_slope_b=-2 * np.sum(y -self.m*X - self.b)
      self.b=self.b-(loss_slope_b*self.lr)
      loss_slope_m=-2 * np.sum((y -self.m*X - self.b)*X)
      self.m=self.m-(loss_slope_m*self.lr)
      # print(loss_slope,self.b)
    print(self.b,self.m)
  def predict(self,X):
    X=X.ravel()
    y_pred=self.m*X + self.b
    return y_pred
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
