# see p.59 QF for dummies
import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1
seed = 4285
days = 1000
np.random.seed(seed)
num_stocks = 6


def random_walk(S_0, N):
    S = S_0
    res = []
    for i in range(1, int(N+1)):
        res.append(S)
        S += np.random.normal(mu, sigma)

    return res


S_0 = 100
rws = []
for _ in range(num_stocks):
    rws.append(random_walk(S_0, days))

xb = np.linspace(1, days, days)
for rw in rws:
    plt.plot(xb, rw)

plt.title('Simulated random walks')
plt.show()
