# K-nearest neighbor is algorithm that classifies based on 
#      logic of classifying testing data based on its distance from
#        from previously clustered training data (which is done via .fit() method).
# K = # of neighbors to consider (how many training points to consider for euclidean distance)
# knowing the label that corresponds to the closest distance will give the classification

import random
from scipy.spatial import distance

def euc(a,b):
    # distance between train and test data 
    return distance.euclidean(a,b)


class KNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict():
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self,row):
        best_distance = euc(row,X_train[0])
        best_index = 0 
        for i in range(1,len(self.X_train)):
            dist = euc(row,X_train[i])
            if dist < best_distance:
                dist = best_distance
                best_index = i
        return self.y_train[best_index]
        

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target



classifier = KNN()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

from sklearn import neighbors as nb
clf = nb.KNeighborsClassifier()
clf.fit(X_train,y_train)

print("Target values: " , y_test)
y_predict = clf.predict(X_test)
print("models predicted values: " , y_predict)

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_predict))
