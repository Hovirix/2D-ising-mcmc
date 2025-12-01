import matplotlib.pyplot as plt
import numpy as np


class IsingModel:
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.beta = 1.0 / T
        self.config = np.random.choice([-1, 1], size=(N, N))

    def cost(self):
        """Computes total energy of the system using vectorized operations."""
        up = np.roll(self.config, -1, axis=0)
        down = np.roll(self.config, 1, axis=0)
        left = np.roll(self.config, -1, axis=1)
        right = np.roll(self.config, 1, axis=1)

        neighbors = up + down + left + right

        return -0.5 * np.sum(self.config * neighbors)

    def delta_cost(self, i, j):
        """Computes energy change for flipping spin at (i, j)."""
        s = self.config[i, j]

        n_up = self.config[(i - 1) % self.N, j]
        n_down = self.config[(i + 1) % self.N, j]
        n_left = self.config[i, (j - 1) % self.N]
        n_right = self.config[i, (j + 1) % self.N]

        return 2 * s * (n_up + n_down + n_left + n_right)

    def step(self):
        """Performs one Metropolis-Hastings step (attempting one flip)."""
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)

        dE = self.delta_cost(i, j)

        if dE <= 0 or np.random.rand() < np.exp(-self.beta * dE):
            self.config[i, j] *= -1
            return True, dE
        return False, 0

    def run_mcmc(self, steps):
        """Runs MCMC for a specific number of steps (flips)."""
        accepted_count = 0
        current_energy = self.cost()

        for _ in range(steps):
            accepted, dE = self.step()
            if accepted:
                accepted_count += 1
                current_energy += dE

        return accepted_count / steps, current_energy

    def display(self, title=None):
        """Visualizes the configuration."""
        plt.imshow(self.config, cmap="Greys", interpolation="nearest")
        if title:
            plt.title(title)
        plt.xticks([])
        plt.yticks([])

    def magnetization(self):
        """Computes magnetization per spin."""
        return np.mean(self.config)
