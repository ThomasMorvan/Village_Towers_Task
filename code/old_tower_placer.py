"""Raw algorithm from paper"""
import numpy as np
import matplotlib.pyplot as plt

apparatus_size = 200  # cm
rwd_density = 7.7  # towers/m
n_rwd_density = 2.3  # towers/m
refractory_period = 12  # cm


L = apparatus_size  # maximum possible location of the tower
dy = refractory_period  # minimum distance between towers
mu_reward = rwd_density  # mean number of towers per meter
mu_no_reward = n_rwd_density  # mean number of towers per meter


def f(L: float, dy: float, mu: float,
      rng: np.random.Generator | None = None) -> np.ndarray:
    """Spatial Poisson process with refractory interval.

    Inputs:
        L : float = Available length. Output locations will be in [0, L].
        dy : float = Minimum spacing between towers.
        mu : float = Tower density.

    Output:
        y : np.ndarray = Sorted tower locations in [0, L]."""

    if rng is None:
        rng = np.random.default_rng()

    if L <= 0:
        raise ValueError("L must be > 0")
    if dy <= 0:
        raise ValueError("dy must be > 0")
    if mu < 0:
        raise ValueError("mu must be >= 0")

    # 1
    maxN = int(np.floor(L / dy))
    print(maxN)

    # 2
    while True:
        n = int(rng.poisson(mu))
        if n <= maxN:
            break

    # 3
    Leffective = L - ((n - 1) * dy)

    # 4
    y = rng.uniform(0.0, 1.0, size=n)

    # 5
    y = np.sort(y)

    # 6
    y = y * Leffective + np.arange(n) * dy

    # 7
    y = y + float(rng.uniform(0.0, L))

    # 8
    y = np.where(y > L, y - L, y)

    # round to nearest integer cm
    y = np.round(y)

    return np.sort(y)


iters = 100000
iters = 1
deltas = []
Rs = []
NRs = []
weird = 0

_deltas = []
_Rs = []
_NRs = []

for i in range(iters):
    R = f(L, dy, mu_reward)
    NR = f(L, dy, mu_no_reward)
    # print("Reward:", R, "No reward towers:", NR)
    delta = abs(len(R) - len(NR))

    if len(NR) > len(R):
        weird += 1
        _Rs.append(len(NR))
        _NRs.append(len(R))
        _deltas.append(delta)
        # R, NR = NR, R  # swap so R is always >= NR
    else:
        _Rs.append(len(R))
        _NRs.append(len(NR))
        _deltas.append(delta)

    Rs.append(len(R))
    NRs.append(len(NR))
    deltas.append(delta)


print(f"weird: {weird} out of {iters} iterations ({weird/iters:.2%})")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
bins = np.arange(0, 20, 1)

axs[0].hist(Rs, bins=bins, density=True)
axs[0].set_xlabel("# Reward Towers")
axs[0].hist(_Rs, bins=bins, density=True, histtype='step', label="weird R", lw=2)

axs[1].hist(NRs, bins=bins, density=True)
axs[1].set_xlabel("# No Reward Towers")
axs[1].hist(_NRs, bins=bins, density=True, histtype='step', label="weird NR", lw=2)

axs[2].hist(deltas, bins=bins, density=True)
axs[2].set_xlabel("Delta (|#R - #NR|)")
axs[2].hist(_deltas, bins=bins, density=True, histtype='step', label="weird delta", lw=2)

for ax in axs:
    ax.set_ylabel("Density")
    ax.set_xlim(0, 20)
    ax.set_xticks(bins)
    ax.set_ylim(0, 0.3)

plt.show()


for i in range(10):
    R = f(L, dy, mu_reward)
    NR = f(L, dy, mu_no_reward)
    print("Reward:", R, "No reward towers:", NR)
