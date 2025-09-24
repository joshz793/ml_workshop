'''
This example illustrates the challenges of fitting a non-linear equation.
Plotting flags are provided for convenience
'''
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from example02_secret_function import secret_model
def safe_print(*args):
    try: 
        print(*args)
    except:
        pass

#%%
np.random.seed(0)

# Parameters for randomly generated data
num_points = 100

# Plotting flags
plot_data = True # Generate plot of training data
plot_best_fit = True # Generate plot of best fit line with training data

# Parameters for training data
train_min_x = 0
train_max_x = 4

# Parameters for testing data
num_test = 1000
test_min_x = 0
test_max_x = 4

# Generate data
x = np.linspace(train_min_x, train_max_x, num_points)
y = secret_model(x)

#%%
if plot_data:
    fig, ax = plt.subplots(1,1)
    ax.scatter(x,y)
    ax.set_xlabel('X')
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig('secret_function_with_noise.png')

#%%
weights = np.ones(1) # TODO: Define number of parameters

### TODO: Define a model
def model(x, weights):
    a = weights[0]
    # b = ...
    return np.zeros(len(x))

# Use the Root Mean Squared Error (RMSE) as the function to minimize
def loss_function(weights):
    return np.sqrt(np.mean((y - model(x, weights))**2))

#%%
results = optimize.minimize(loss_function, weights)
weights = results['x']
loss = results['fun']
safe_print("Weights:\t", weights)
safe_print(f"Final Training Error:\t{loss:.6f}")

#%%
if plot_best_fit:
    fig, ax = plt.subplots(1,1)
    ax.scatter(x,y)
    ax.plot(x, model(x,weights), 'r', label='Best Fit Curve')
    ax.set_xlabel('X')
    ax.set_ylabel("Y")
    ax.legend()
    fig.tight_layout()
    fig.savefig('nonlinear_best_fit_curve.png')

#%%
x = (test_max_x - test_min_x) * np.random.random_sample(num_test) - test_min_x
y = secret_model(x)
test_error = np.sqrt(np.mean((y - model(x,weights))**2))
safe_print(f'Final Testing Error:\t{test_error:.6f}')
