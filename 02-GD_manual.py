import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value

# our model forward pass


def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) **2 


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


w_list = []
l_list = []

# Before training
print("predict (before training)",  'x=4','y=', forward(4))

# Training loop

for epoch in range(100):
    
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.001 * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
        l_sum += l


    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l_sum, 2))
    w_list.append(w)
    l_list.append(l_sum/len(y_data))
    
# After training
print("predict (after training)",  "x=4", 'y=', forward(4))


plt.plot(w_list, l_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()