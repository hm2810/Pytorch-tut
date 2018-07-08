import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Special variable with random value

# our model forward pass


def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) **2 


w_list = []
l_list = []

# Before training
print("predict (before training)",  'x=4','y=', forward(4).data[0])

# Training loop

for epoch in range(100):
    
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()    #Special function for BP
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        l_sum += l
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()


    print("progress:", epoch, "w=", w.data, "loss=", l_sum)
    w_list.append(w.data[0])
    l_list.append(l_sum/len(y_data))
    
# After training
print("predict (after training)",  "x=4", 'y=', forward(4).data[0])


plt.plot(w_list, l_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()