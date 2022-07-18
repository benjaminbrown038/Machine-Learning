# working productively requires addressing the knowledge that
#	 I already KNOW 

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# will affect accuracy of model because of how randomly the train, test data 
#	partitioned
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

# from sklearn import tree
# classify = tree.DecisionTreeClassifier()
# classify.fit(X_train,y_train)


from sklearn import neighbors as nb
clf = nb.KNeighborsClassifier()
clf.fit(X_train,y_train)

print("Target values: " , y_test)
y_predict = clf.predict(X_test)
print("models predicted values: " , y_predict)

# make the decision and take action to walk out the door
#	as did walk in the door

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_predict))
