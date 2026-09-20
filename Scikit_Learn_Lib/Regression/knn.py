from sklearn.neighbors import KNeighborsClassifier

x = [
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [330,9],
    [360,9.5]
]
y=[0,0,0,1,1,1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
a = float(input("Enter weight in grams"))
b = float(input("Enter size in cm"))
predict = model.predict([[a,b]])[0]

if predict==0:
    print("Orange")
else:
    print("Apple")