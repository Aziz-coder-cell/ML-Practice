from sklearn.linear_model import LogisticRegression

Hours = [[1], [2], [3], [4], [5], [6]]
Result = [0, 0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(Hours,Result)

print(model.predict([[3.4]])) 