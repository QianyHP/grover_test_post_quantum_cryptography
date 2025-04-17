import numpy as np
import matplotlib.pyplot as plt

def grover_best_iteration(theta):
    """Returns the best integer number of Grover iterations r* and its corresponding success probability."""
    r_float = (np.pi / (4 * theta)) - 0.5
    r_floor = int(np.floor(r_float))
    r_ceil = int(np.ceil(r_float))

    p_floor = np.sin((2 * r_floor + 1) * theta) ** 2
    p_ceil = np.sin((2 * r_ceil + 1) * theta) ** 2

    return (r_floor, p_floor) if p_floor >= p_ceil else (r_ceil, p_ceil)

def plot_grover_success_range(n_min=3, n_max=8):
    plt.figure(figsize=(10, 3.5))
    print(f"{'Qubits':>6} | {'Search Space N':>15} | {'Best Iteration':>15} | {'Max Success Probability':>25}")
    print("-" * 75)

    line_styles = ['-', '--', '-.', ':']  # 线型轮换

    for idx, n_qubits in enumerate(range(n_min, n_max + 1)):
        N = 2 ** n_qubits
        theta = np.arcsin(1 / np.sqrt(N))
        iterations = np.arange(1, 21)
        probs = np.sin((2 * iterations + 1) * theta) ** 2

        best_iter, best_prob = grover_best_iteration(theta)
        print(f"{n_qubits:>6} | {N:>15} | {best_iter:>15} | {best_prob:>25.5f}")

        # 设置线型、线宽（随比特数增大）和透明度
        style = line_styles[idx % len(line_styles)]
        linewidth = 1.5 + 0.2 * idx  # 线宽逐步增加
        plt.plot(iterations, probs,
                 label=f"n={n_qubits} (best r={best_iter})",
                 linestyle=style,
                 linewidth=linewidth,
                 alpha=0.85)

    plt.xlabel("Number of Grover Iterations")
    plt.ylabel("Success Probability")
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(1, 21))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig("grover_success_range.png", dpi=300)
    plt.show()
plot_grover_success_range(n_min=3, n_max=8)
