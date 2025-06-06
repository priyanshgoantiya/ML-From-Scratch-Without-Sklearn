## multiple_linear_regression
import numpy 
class multiple_linear_regression:
  def __init__(self):
    self.coef=None
    self.intercept=None
  def fit(self,X_train,y_train):
    X_train=np.insert(X_train,0,1,axis=1)
    # clac all coeffs
    betas=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
    self.intercept=betas[0]
    self.coef=betas[1:]
  def predict(self,X_test):
    y_pred= np.dot(X_test,self.coef) + self.intercept
    return y_pred
  def r2_score(self, y_test, y_pred):
    ssr = np.sum((y_test - y_pred) ** 2)
    ssm = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ssr / ssm)
  def adjusted_r2_score(self,y_true, y_pred, p):
    r2 = self.r2_score(y_true, y_pred)
    n=len(X_test)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adj_r2
