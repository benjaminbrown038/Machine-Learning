from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

# set of rules
classifier = tree.DecisionTreeClassifier()
# algorithm that sets the rules 
classifier = classifier.fit(features,labels)
print(classifier.predict([[165,0]]))
