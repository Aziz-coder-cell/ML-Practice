import torch

x= torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
learning_rate = 0.1

y_pred = w * x 
y_true = 12 

loss = (y_pred - y_true) **2
loss.backward()

print(loss.item()) 
print(w.grad.item()) 

with torch.no_grad(): 
    w -= learning_rate * w.grad 

w.grad.zero_() 
print(w.item()) 