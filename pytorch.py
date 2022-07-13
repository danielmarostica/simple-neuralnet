import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # neural networks
torch.set_num_threads(8) # multiprocessing

# choose learning rate
learning_rate = 0.1


# create fake data
x = np.arange(-1, 1, 0.01)
y = x ** 3


# transform data into float32 tensors
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()


# choose an activation function
activation_func = nn.Softplus()

# pre-nodes transformation (y = a * W + b)
n_input = len(x) # input size
n_output = 3 # number of neurons
linear_transformation_1 = nn.Linear(n_input, n_output)

# post-nodes transformation
n_input = 3 # number of neurons
n_output = 200 # output_size
linear_transformation_2 = nn.Linear(n_input, n_output) # includes random weights and biases


# define model
model = nn.Sequential(
    linear_transformation_1,
    activation_func,
    linear_transformation_2
)


loss_function = nn.L1Loss() # mean absolute error (couldn't find sum of squared errors to mimmick the "from_scratch" version)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # makes gradient descent even better and faster


loss_history = []
epochs = 5000
for epoch in range(epochs):
    
    # forward propagation
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    loss_history.append(loss.item())

    # backward propagation
    model.zero_grad() # reset gradient of previous loop
    
    # calculate new gradients
    loss.backward()
    
    # find global minima
    optimizer.step()


with torch.no_grad(): # get rid of gradients
    y_pred = model(x)


# plot result
plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Data')
plt.plot(x, y_pred, '--', label='Neural Network')
plt.legend(frameon=False)
plt.savefig('pytorch_output.jpg')

# plot loss
plt.figure(figsize=(10, 10))
plt.plot(range(epochs), loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('pytorch_loss.jpg')