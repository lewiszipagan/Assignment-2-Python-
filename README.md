"""
Created on Sat May 4 16:41:41 2024
@author: Walther Lewis Zipagan
"""
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generate some sample data
n = 100
x = np.random.normal(0, 1, n)
y = 2 * x + 1 + np.random.normal(0, 1, n)

# Create the PyMC3 model
with pm.Model() as model:
    # Specify the priors
    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 10)
    sigma = pm.HalfNormal('sigma', 5)

    # Define the likelihood
    y_pred = alpha + beta * x
    likelihood = pm.Normal('likelihood', y_pred, sigma, observed=y)

    # Perform Bayesian inference
    trace = pm.sample(2000, tune=1000, chains=2)

# Analyze the results
pm.summary(trace)

# Plot the posterior distributions
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pm.plot_posterior(trace, ax=ax)
plt.show()
