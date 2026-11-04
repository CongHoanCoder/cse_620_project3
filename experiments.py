"""Run all experiments required by the CSE620 Simulated Annealing mini-project.

Sections 1-3 of CSE620_SA_Project_Notebook.ipynb are reproduced exactly; this
script only automates the sweep over the parameters asked for in the report.

Outputs (written to ./results):
    section1_exponential.csv
    section2_linear.csv
    section3_sa_vs_hc.csv
    figures used by the LaTeX report
"""

import copy
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Problem setup (identical to the notebook)
# --------------------------------------------------------------------------
map_img = mpimg.imread("map.png")

with open("capitals.json", "r") as capitals_file:
    capitals = json.load(capitals_file)
capitals_list = list(capitals.items())


def show_path(path, starting_city, filename, w=12, h=8):
    x, y = list(zip(*path))
    _, (x0, y0) = starting_city
    plt.imshow(map_img)
    plt.plot(x0, y0, "y*", markersize=15)
    plt.plot(x + x[:1], y + y[:1])
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def L2_dist(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5


def L1_dist(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


class TravelingSalesmanProblem:
    def __init__(self, cities, dist_measure=L1_dist):
        self.path = copy.deepcopy(cities)
        self.dist_measure = dist_measure

    def copy(self):
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp

    @property
    def names(self):
        names, _ = zip(*self.path)
        return names

    @property
    def coords(self):
        _, coords = zip(*self.path)
        return coords

    def neighborhood(self):
        neighbors = []
        for i in range(len(self.path) - 1):
            new_path = self.copy()
            new_path.path[i], new_path.path[i + 1] = new_path.path[i + 1], new_path.path[i]
            neighbors.append(new_path)

        last_path = self.copy()
        last_path.path[0], last_path.path[-1] = last_path.path[-1], last_path.path[0]
        neighbors.append(last_path)

        return neighbors

    def evaluate(self):
        length = 0
        coords = self.coords
        for i in range(len(coords) - 1):
            length += np.abs(self.dist_measure(coords[i], coords[i + 1]))
        length += np.abs(self.dist_measure(coords[0], coords[-1]))
        return length


def generate_TSP(num_cities, random_state=None):
    rand_gen = np.random.default_rng(random_state)
    capitals_idx = rand_gen.choice(
        list(range(0, len(capitals_list))), size=(num_cities,), replace=False
    )
    return np.array(capitals_list, dtype="object")[capitals_idx].tolist()


# --------------------------------------------------------------------------
# Optimizers (identical to the notebook)
# --------------------------------------------------------------------------
class Schedule:
    def __init__(self, method="exponential", alpha=0.95, starting_temp=100):
        self.method = method
        self.alpha = alpha
        self.T_0 = starting_temp

    def cool(self, k):
        if self.method == "exponential":
            return self.T_0 * (self.alpha ** k)
        if self.method == "logarithmic":
            return self.T_0 / (1 + self.alpha * np.log(1 + k))
        if self.method == "linear":
            return self.T_0 / (1 + self.alpha * k)
        raise NotImplementedError(f"There is no cooling schedule matching {self.method}")


def simulated_annealing(
    problem, cooling_schedule, stop_temp=1e-5, max_iter=np.inf, random_state=None, verbose=True
):
    rand_gen = np.random.default_rng(random_state)

    current = problem
    best_solution = current
    best_cost = current.evaluate()

    T_k = cooling_schedule.T_0
    k = 0

    while k < max_iter and T_k > stop_temp:
        T_k = cooling_schedule.cool(k)

        next_state = rand_gen.choice(current.neighborhood())

        current_cost = current.evaluate()
        next_cost = next_state.evaluate()

        if next_cost < current_cost:
            current = next_state
            if next_cost < best_cost:
                best_solution = next_state
                best_cost = next_cost
        else:
            prob = np.exp((current_cost - next_cost) / T_k)
            R = rand_gen.uniform(0, 1)
            if prob > R:
                current = next_state

        k += 1

    if verbose:
        print(f"Terminated with temperature T={T_k:.2g}\nAt iteration #{k:,}.")

    return best_solution


def hill_climbing(problem, max_iter=1e10, min_delta=0):
    iterations = 0
    current = problem
    delta = np.inf

    while delta > min_delta and iterations < max_iter:
        neighborhood = current.neighborhood()
        best_neighbor_idx = np.argmin([x.evaluate() for x in neighborhood])
        next_state = neighborhood[best_neighbor_idx]
        delta = current.evaluate() - next_state.evaluate()
        current = next_state
        iterations += 1
    return current


# --------------------------------------------------------------------------
# Experiment drivers
# --------------------------------------------------------------------------
N_REPEAT = 10
SEED_SA = 192
SEED_CITIES = 254
ALPHAS = [0.95, 0.60, 0.30]
START_TEMPS = [4000, 2000]


def timed_sa(problem, schedule, stop_temp=1e-5, max_iter=np.inf):
    """Average wall-clock time (ms) over N_REPEAT seeded runs + the best cost."""
    avg_time = 0.0
    result = None
    for _ in range(N_REPEAT):
        start = time.perf_counter()
        result = simulated_annealing(
            problem,
            schedule,
            stop_temp=stop_temp,
            max_iter=max_iter,
            random_state=SEED_SA,
            verbose=False,
        )
        avg_time += (time.perf_counter() - start) * 1e3 / N_REPEAT
    return avg_time, result.evaluate()


def timed_hc(problem):
    avg_time = 0.0
    result = None
    for _ in range(N_REPEAT):
        start = time.perf_counter()
        result = hill_climbing(problem)
        avg_time += (time.perf_counter() - start) * 1e3 / N_REPEAT
    return avg_time, result.evaluate()


def section1():
    """Exponential decay sweep (notebook Section 1: stop_temp=1e-5, max_iter=inf)."""
    subset = generate_TSP(num_cities=15, random_state=SEED_CITIES)
    tsp = TravelingSalesmanProblem(subset)
    rows = []
    for T0 in START_TEMPS:
        for alpha in ALPHAS:
            schedule = Schedule(method="exponential", alpha=alpha, starting_temp=T0)
            t, cost = timed_sa(tsp, schedule, stop_temp=1e-5, max_iter=np.inf)
            rows.append(
                {"Starting Temperature": T0, "Alpha": alpha,
                 "Average Time (ms)": round(t, 2), "Best Cost": round(cost, 1)}
            )
            print(f"exp  T0={T0} alpha={alpha}: {t:.2f} ms, cost={cost:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/section1_exponential.csv", index=False)
    return df


def section2():
    """Linear decay sweep (notebook Section 2: stop_temp=1, max_iter=1e10)."""
    subset = generate_TSP(num_cities=15, random_state=SEED_CITIES)
    tsp = TravelingSalesmanProblem(subset)
    rows = []
    for T0 in START_TEMPS:
        for alpha in ALPHAS:
            schedule = Schedule(method="linear", alpha=alpha, starting_temp=T0)
            t, cost = timed_sa(tsp, schedule, stop_temp=1, max_iter=1e10)
            rows.append(
                {"Starting Temperature": T0, "Alpha": alpha,
                 "Average Time (ms)": round(t, 2), "Best Cost": round(cost, 1)}
            )
            print(f"lin  T0={T0} alpha={alpha}: {t:.2f} ms, cost={cost:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/section2_linear.csv", index=False)
    return df


def section3(best_method, best_alpha, best_T0):
    """SA (best parameters) vs. hill climbing for 5, 20 and 30 cities."""
    rows = []
    for n_cities in [5, 20, 30]:
        subset = generate_TSP(num_cities=n_cities, random_state=SEED_CITIES)
        tsp = TravelingSalesmanProblem(subset)

        schedule = Schedule(method=best_method, alpha=best_alpha, starting_temp=best_T0)
        # Notebook Section 3 stops at the default 1e-5 for exponential schedules;
        # the linear sweeps of Section 2 stop at 1 (1e-5 would need ~1e8 iterations).
        stop_temp = 1e-5 if best_method == "exponential" else 1
        sa_time, sa_cost = timed_sa(tsp, schedule, stop_temp=stop_temp, max_iter=np.inf)
        hc_time, hc_cost = timed_hc(tsp)
        rows.append(
            {"# Cities": n_cities,
             "SA Time (ms)": round(sa_time, 2), "SA Cost": round(sa_cost, 1),
             "HC Time (ms)": round(hc_time, 2), "HC Cost": round(hc_cost, 1)}
        )
        print(
            f"n={n_cities}: SA {sa_time:.2f} ms / {sa_cost:.1f} | "
            f"HC {hc_time:.2f} ms / {hc_cost:.1f}",
            flush=True,
        )

        # Save the best SA tour of the largest instance for the report figure.
        if n_cities == 30:
            result = simulated_annealing(
                tsp, schedule, stop_temp=stop_temp, max_iter=np.inf,
                random_state=SEED_SA, verbose=False,
            )
            show_path(result.coords, subset[0], "results/sa_path_30.png")

    df = pd.DataFrame(rows)
    df.to_csv("results/section3_sa_vs_hc.csv", index=False)
    return df


def cooling_curves():
    """Figure: temperature vs. iteration for the six exponential schedules.

    Curves are clipped at the 1e-5 stop temperature used in Section 1; the
    crossing point of each curve *is* the number of iterations that schedule
    performs (k_max = ln(stop/T0)/ln(alpha))."""
    stop_temp = 1e-5
    ks = np.arange(0, 450)
    plt.figure(figsize=(6.5, 4))
    kmax_lines = []
    for T0 in START_TEMPS:
        row = []
        for alpha in ALPHAS:
            temp = T0 * alpha ** ks
            plt.plot(ks, temp, label=rf"$T_0={T0}$, $\alpha={alpha}$")
            k_max = int(np.argmax(temp <= stop_temp))
            row.append(str(k_max))
            if temp[-1] <= stop_temp:
                plt.plot([k_max], [stop_temp], "o", ms=4)
        kmax_lines.append(
            f"$T_0={T0}$:  " + ",  ".join(
                rf"$\alpha={a}$ $\to$ {k}" for a, k in zip(ALPHAS, row))
        )
    plt.yscale("log")
    plt.ylim(stop_temp / 3, 1e4)
    plt.xlabel("Iteration $k$")
    plt.ylabel(r"Temperature $T_k = T_0\alpha^k$")
    plt.title("Exponential cooling schedules (until the $10^{-5}$ stop temperature)")
    plt.legend(fontsize=8, ncol=2, loc="upper right")
    plt.grid(alpha=0.3, which="both")
    plt.text(
        0.13, 0.06,
        "Iterations until $T<10^{-5}$ ($k_{max}$):\n" + "\n".join(kmax_lines),
        transform=plt.gca().transAxes, fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig("results/cooling_curves.png", dpi=150)
    plt.close()


def main():
    import os

    os.makedirs("results", exist_ok=True)
    print("=== Section 1: exponential decay ===", flush=True)
    s1 = section1()
    print(s1, flush=True)

    print("\n=== Section 2: linear decay ===", flush=True)
    s2 = section2()
    print(s2, flush=True)

    # Best parameter combination over both tables = lowest best cost.
    combined = pd.concat(
        [s1.assign(schedule="exponential"), s2.assign(schedule="linear")],
        ignore_index=True,
    )
    best = combined.loc[combined["Best Cost"].idxmin()]
    print("\nBest parameter combination:", best.to_dict(), flush=True)

    print("\n=== Section 3: SA vs. hill climbing ===", flush=True)
    s3 = section3(best["schedule"], best["Alpha"], best["Starting Temperature"])
    print(s3, flush=True)

    cooling_curves()
    print("\nDone. Results in ./results", flush=True)


if __name__ == "__main__":
    main()
