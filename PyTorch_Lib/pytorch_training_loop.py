import torch

x = torch.tensor([1.0,2.0,3.0,4.0])
y_true = torch.tensor([2.0,4.0,6.0,8.0])

w = torch.tensor(0.0,requires_grad=True)

learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    #Step-1
    y_pred = w * x

    #Step-2
    loss = ((y_pred-y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    print(f"Epoch {epoch+1}: w={w.item():.4f}, loss={loss.item():.4f}")

print(f"Y_True:{y_true},Y_Predict:{y_pred}")