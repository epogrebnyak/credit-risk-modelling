"""
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 2. In 
this example, we use the functions in binomialPoissonModels.py to simulate
the binomial and Poisson independent-default models, print the results and
graph the tail probabilities.
-----------------
David Jamieson Bolder, February 2018
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylab

pylab.ion()
pylab.show()
import seaborn as sns

sns.set()
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = Path(__file__).parent / "data"
# These are the exposure and default-probability files
dpFile = myHome / "defaultProbabilties.npy"
expFile = myHome / "exposures.npy"
import binomialPoissonModels as bp

plt.close("all")
# Key inputs and parameters
c = np.load(expFile)
p = np.load(dpFile)
N = len(c)
myRho = 0.05
portfolioSize = np.sum(c)
myC = portfolioSize / N
M = 1000000
numberOfModels = 2
alpha = np.array([0.95, 0.97, 0.99, 0.995, 0.999, 0.9997, 0.9999])
myP = np.mean(p)
myC = np.mean(c)
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha), numberOfModels])
es = np.zeros([len(alpha), numberOfModels])
cTime = np.zeros(numberOfModels)
# Binomial model
startTime = time.perf_counter()
el[0], ul[0], var[:, 0], es[:, 0] = bp.independentBinomialSimulation(N, M, p, c, alpha)
cTime[0] = time.perf_counter() - startTime
# Poisson model
startTime = time.perf_counter()
el[1], ul[1], var[:, 1], es[:, 1] = bp.independentPoissonSimulation(N, M, p, c, alpha)
cTime[1] = time.perf_counter() - startTime
# =====================
# TABLE: Key Model results
# =====================
print("Alpha\t VaR_b\t ES_b\t VaR_p\t ES_p")
for n in range(0, len(alpha)):
    print(
        "%0.2fth\t %0.1f\t %0.1f\t %0.1f\t %0.1f"
        % (1e2 * alpha[n], var[n, 0], es[n, 0], var[n, 1], es[n, 1])
    )
print("Expected loss: %0.1f vs. %0.1f" % (el[0], el[1]))
print("Loss volatility: %0.1f vs. %0.1f" % (ul[0], ul[1]))
print("CPU Time: %0.1f vs. %0.1f" % (cTime[0], cTime[1]))
# =====================
plt.figure(1)  # Plot the independent default simulation results
# =====================
plt.plot(var[:, 0], alpha, color="red", linestyle="-", label="Binomial")
plt.plot(var[:, 1], alpha, color="blue", linestyle="--", label="Poisson")
plt.xlabel("USD")
plt.ylabel("Quantile")
plt.legend(loc=4)