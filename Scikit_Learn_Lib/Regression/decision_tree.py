from sklearn.tree import DecisionTreeClassifier

X = [
    [7,2],
    [8,3],
    [9,8],
    [10,9]
]
y = [0,0,1,1]

model = DecisionTreeClassifier()
model.fit(X,y)

print(model.predict([[8.5768,4]])[0])