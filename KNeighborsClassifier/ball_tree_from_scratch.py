##ball_tree_from_scratch
from sklean.neigbours import BallTree
import numpy 
class kd_tree_from_scratch:
  def __init__(self,K):
    self.neighbors=K
    self.X_train=None
    self.y_train=None
    self.tree=None
  def fit(self,X_train,y_train):
    self.X_train=np.array(X_train)
    self.y_train=np.array(y_train)
    self.tree=BallTree(X_train)
  def predict(self,X_test):
    prediction=[]
    X_test = np.array(X_test)
    for i in range(len(X_test)):
      distances,indices=self.tree.query(X_test[i],k=self.neighbors)
      majority_count=np.bincount(self.y_train[indices]).argmax()
      prediction.append(majority_count)
    return prediction
