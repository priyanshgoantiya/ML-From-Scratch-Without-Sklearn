## KNeighborsClassifier_from_scratch
class KNeighborsClassifier_from_scratch:
  def __init__(self,K):
    self.neighbors=K
    self.X_train=None
    self.y_train=None
  def fit(self,X_train,y_train):
    self.X_train=np.array(X_train)
    self.y_train=np.array(y_train)
  def predict(self,X_test):
    prediction=[]
    for i in X_test:
      distances=[]
      for idx,j in enumerate(self.X_train):
        distances.append((idx,self.distance_calc(i,j)))
      neighbors=sorted(distances, key=lambda x: x[1])[0:self.neighbors]
      neighbors_indices=[idx for idx,_ in neighbors]
      majority_count=np.bincount(self.y_train[neighbors_indices]).argmax()
      prediction.append(majority_count)
    return prediction
  def distance_calc(self,i,j):
    return np.linalg.norm(i-j)
