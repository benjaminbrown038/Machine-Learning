from sklearn.datasets import load_iris
import numpy as np

# loads and returns the iris dataset
iris = load_iris()

test_index = [0,50,100]

# train
train_data = np.delete(iris.data,test_index)
train_target = np.delete(iris.target,test_index)

# test
test_data = iris.data[test_index]
test_target = iris.target[test_index]

from sklearn import tree
clf = tree.DecisionTreeClassifier()


clf.fit(iris.data,iris.target)
print(test_target)
print("model prediction" + "\n")
print(clf.predict(test_data))

