from sklearn.linear_model import LinearRegression

x = [[4],[5],[3],[3.5],[6],[2.7]]
y = [[43],[47],[37],[39],[50],[25]]

model = LinearRegression()
model.fit(x,y)

a = float(input('Enter the hours studied'))

result = model.predict([[a]])

print(result)