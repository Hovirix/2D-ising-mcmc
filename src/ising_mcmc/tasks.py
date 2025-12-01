import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .clusters import find_clusters
from .ising_model import IsingModel

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def task_2_parameters():
    print("--- Running Task 2: Exploring MCMC Parameters ---")
    Ts = [0.5, 2.5, 5.0]
    Ns = [10, 50, 100]

    plt.figure(figsize=(10, 8))
    plot_idx = 1
    for N in Ns:
        for T in Ts:
            model = IsingModel(N, T)
            model.run_mcmc(steps=N * N * 100)

            plt.subplot(3, 3, plot_idx)
            model.display(f"N={N}, T={T}")
            plot_idx += 1
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_snapshots.png")
    plt.close()
    print("Saved output/fig_snapshots.png")

    N_fixed = 50
    burn_in_steps = 100000

    acc_rates = []
    temperatures = np.linspace(0.1, 6.0, 20)

    costs_over_time = {}

    for T in temperatures:
        model = IsingModel(N_fixed, T)
        costs = []
        rate_accum = 0

        chunk_size = 1000
        for _ in range(burn_in_steps // chunk_size):
            r, e = model.run_mcmc(chunk_size)
            costs.append(e)
            rate_accum += r

        acc_rates.append(rate_accum / (burn_in_steps // chunk_size))

        if T in [1.03, 2.27, 4.13]:
            costs_over_time[T] = costs

    plt.figure()
    plt.plot(temperatures, acc_rates, "o-")
    plt.xlabel("Temperature T")
    plt.ylabel("Acceptance Rate")
    plt.title(f"Acceptance Rate vs T (N={N_fixed})")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "fig_acceptance.png")
    plt.close()

    plt.figure()
    for T in [temperatures[2], temperatures[8], temperatures[-3]]:
        model = IsingModel(N_fixed, T)
        costs = []
        for i in range(200):
            _, e = model.run_mcmc(500)
            costs.append(e)
        plt.plot(costs, label=f"T={T:.2f}")
    plt.xlabel("MCMC Time (x500 steps)")
    plt.ylabel("Energy")
    plt.title("Energy convergence (Burn-in check)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fig_equilibration.png")
    plt.close()
    print("Saved output/fig_acceptance.png and output/fig_equilibration.png")


def task_3_critical_temp():
    print("--- Running Task 3: Statistical Properties & Tc ---")
    N = 40
    T_range = np.sort(
        np.concatenate([
            np.linspace(1.0, 2.0, 5),
            np.linspace(2.05, 2.5, 15),
            np.linspace(2.6, 4.0, 5),
        ])
    )

    samples_per_T = 50
    wait_steps = N * N
    burn_in = N * N * 500

    magnetizations = []
    susceptibilities = []

    for T in T_range:
        model = IsingModel(N, T)
        model.run_mcmc(burn_in)

        m_samples = []
        m_sq_samples = []

        for _ in range(samples_per_T):
            model.run_mcmc(wait_steps)
            m = model.magnetization()
            m_samples.append(abs(m))
            m_sq_samples.append(m**2)

        avg_abs_m = np.mean(m_samples)
        avg_m_sq = np.mean(m_sq_samples)

        magnetizations.append(avg_abs_m)

        chi = (1 / T) * (N**2) * (avg_m_sq - avg_abs_m**2)
        susceptibilities.append(chi)

    magnetizations = np.array(magnetizations)
    susceptibilities = np.array(susceptibilities)

    peak_idx = np.argmax(susceptibilities)
    Tc_est = T_range[peak_idx]
    print(f"Estimated Tc from Susceptibility peak: {Tc_est:.3f}")

    plt.figure()
    plt.plot(T_range, magnetizations, "o-", label="<|m|>")
    plt.axvline(x=2.269, color="r", linestyle="--", label="Theoretical Tc")
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization <|m|>")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fig_magnetization.png")
    plt.close()

    plt.figure()
    plt.plot(T_range, susceptibilities, "s-", color="orange", label="Chi")
    plt.axvline(x=Tc_est, color="k", linestyle="--", label=f"Est Tc={Tc_est:.2f}")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Susceptibility $\chi$")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fig_susceptibility.png")
    plt.close()

    mask_low = (T_range < Tc_est) & (T_range > 1.5)
    if np.sum(mask_low) > 3:
        x_data = (Tc_est - T_range[mask_low]) / Tc_est
        y_data = magnetizations[mask_low]

        log_x = np.log(x_data)
        log_y = np.log(y_data)

        slope_m, intercept_m = np.polyfit(log_x, log_y, 1)
        print(f"Estimated beta exponent: {slope_m:.3f} (Theoretical 0.125)")

        plt.figure()
        plt.scatter(log_x, log_y)
        plt.plot(
            log_x, slope_m * log_x + intercept_m, "r-", label=f"Slope={slope_m:.2f}"
        )
        plt.xlabel("log((Tc-T)/Tc)")
        plt.ylabel("log(<|m|>)")
        plt.title(f"Critical Exponent Beta Fit (T < {Tc_est})")
        plt.legend()
        plt.savefig(OUTPUT_DIR / "fig_fit_beta.png")
        plt.close()

    mask_high = (T_range > Tc_est) & (T_range < 3.5)
    if np.sum(mask_high) > 3:
        x_data = (T_range[mask_high] - Tc_est) / Tc_est
        y_data = susceptibilities[mask_high]

        log_x = np.log(x_data)
        log_y = np.log(y_data)

        slope_chi, intercept_chi = np.polyfit(log_x, log_y, 1)
        gamma_est = -slope_chi
        print(f"Estimated gamma exponent: {gamma_est:.3f} (Theoretical 1.75)")

        plt.figure()
        plt.scatter(log_x, log_y)
        plt.plot(
            log_x,
            slope_chi * log_x + intercept_chi,
            "r-",
            label=f"Slope={slope_chi:.2f}",
        )
        plt.xlabel("log((T-Tc)/Tc)")
        plt.ylabel("log(Chi)")
        plt.title(f"Critical Exponent Gamma Fit (T > {Tc_est})")
        plt.legend()
        plt.savefig(OUTPUT_DIR / "fig_fit_gamma.png")
        plt.close()


def task_4_clusters():
    print("--- Running Task 4: Cluster Sizes ---")
    N = 50
    Tc = 2.27
    Ts = [1.5, Tc, 3.5]

    plt.figure(figsize=(12, 4))

    for i, T in enumerate(Ts):
        model = IsingModel(N, T)
        model.run_mcmc(steps=N * N * 500)

        sizes = find_clusters(model.config)

        plt.subplot(1, 3, i + 1)
        plt.hist(sizes, bins=30, log=True, color="purple", alpha=0.7)
        plt.title(f"Clusters T={T}")
        plt.xlabel("Cluster Size")
        plt.ylabel("Frequency (Log Scale)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_clusters.png")
    plt.close()
    print("Saved output/fig_clusters.png")
