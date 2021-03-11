import numpy as np
import matplotlib.pyplot as plt
import random

# choose learning rate
learning_rate = 0.001

# create fake data
x = np.arange(-1, 1, 0.01)
y_func = lambda x: x**3
y = y_func(x)

# initialize weights & biases
w1 = random.normalvariate(0, 1)
w2 = random.normalvariate(0, 1)
w3 = random.normalvariate(0, 1)
w4 = random.normalvariate(0, 1)
w5 = random.normalvariate(0, 1)
w6 = random.normalvariate(0, 1)

b1 = 0
b2 = 0
b3 = 0
b4 = 0

# choose an activation function
activation_func = lambda x: np.log(1 + np.e ** x)

# pre-node function
f1 = lambda x: w1*x + b1
f2 = lambda x: w2*x + b2
f3 = lambda x: w3*x + b3

# first run through network
y_pred = activation_func(f1(x)) * w4 + activation_func(f2(x)) * w5 + activation_func(f3(x)) * w6 + b4

epochs = 10000
for epoch in range(epochs):

    # gradient descent
    dSSR_w1 = (-2 * (y - y_pred) * w4 * x * np.e ** f1(x) / (1 + np.e ** f1(x))).sum()
    dSSR_w2 = (-2 * (y - y_pred) * w5 * x * np.e ** f2(x) / (1 + np.e ** f2(x))).sum()
    dSSR_w3 = (-2 * (y - y_pred) * w6 * x * np.e ** f3(x) / (1 + np.e ** f3(x))).sum()
    dSSR_w4 = (-2 * (y - y_pred) * activation_func(f1(x))).sum()
    dSSR_w5 = (-2 * (y - y_pred) * activation_func(f2(x))).sum()
    dSSR_w6 = (-2 * (y - y_pred) * activation_func(f3(x))).sum()

    dSSR_b1 = (-2 * (y - y_pred) * w4 * np.e ** f1(x) / (1 + np.e ** f1(x))).sum()
    dSSR_b2 = (-2 * (y - y_pred) * w5 * np.e ** f2(x) / (1 + np.e ** f2(x))).sum()
    dSSR_b3 = (-2 * (y - y_pred) * w6 * np.e ** f3(x) / (1 + np.e ** f3(x))).sum()
    dSSR_b4 = (-2 * (y - y_pred)).sum()

    # step sizes
    step_size_dSSR_w1 = dSSR_w1 * learning_rate
    step_size_dSSR_w2 = dSSR_w2 * learning_rate
    step_size_dSSR_w3 = dSSR_w3 * learning_rate
    step_size_dSSR_w4 = dSSR_w4 * learning_rate
    step_size_dSSR_w5 = dSSR_w5 * learning_rate
    step_size_dSSR_w6 = dSSR_w6 * learning_rate

    step_size_dSSR_b1 = dSSR_b1 * learning_rate
    step_size_dSSR_b2 = dSSR_b2 * learning_rate
    step_size_dSSR_b3 = dSSR_b3 * learning_rate
    step_size_dSSR_b4 = dSSR_b4 * learning_rate

    # updating weights & biases
    w1 = w1 - step_size_dSSR_w1
    w2 = w2 - step_size_dSSR_w2
    w3 = w3 - step_size_dSSR_w3
    w4 = w4 - step_size_dSSR_w4
    w5 = w5 - step_size_dSSR_w5
    w6 = w6 - step_size_dSSR_w6

    b1 = b1 - step_size_dSSR_b1
    b2 = b2 - step_size_dSSR_b2
    b3 = b3 - step_size_dSSR_b3
    b4 = b4 - step_size_dSSR_b4
    
    # run values through network
    y_pred = activation_func(f1(x)) * w4 + activation_func(f2(x)) * w5 + activation_func(f3(x)) * w6 + b4

# plot result
plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Data')
plt.plot(x, y_pred, '--', label='Neural Network')
plt.legend(frameon=False)