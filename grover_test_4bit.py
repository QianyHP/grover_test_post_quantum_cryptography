from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit import QuantumCircuit, transpile
from qiskit_algorithms import Grover, AmplificationProblem
from qiskit.visualization import plot_distribution
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np


def logic_state_to_qiskit_bitstr(target_state):
    """反转逻辑目标状态，使其符合 Qiskit 的低位在前的测量顺序。"""
    return target_state[::-1]


def build_manual_oracle_4bit(target_state):
    """构造适用于 4 比特的 Oracle，相位反转指定目标态。"""
    oracle = QuantumCircuit(4)
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            oracle.x(i)
    oracle.h(3)
    oracle.mcx([0, 1, 2], 3)  # 控制前三位，翻转最后一位
    oracle.h(3)
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            oracle.x(i)
    return oracle.to_gate(label="PhaseOracle")


def adaptive_grover_search_manual_oracle(target_state="1010", max_iter=6, threshold=0.975,
                                         shots=1024, optimization_level=3, backend_name='qasm_simulator'):
    target_bitstr = logic_state_to_qiskit_bitstr(target_state)
    oracle_gate = build_manual_oracle_4bit(target_state)
    oracle_circuit = QuantumCircuit(4)
    oracle_circuit.append(oracle_gate, [0, 1, 2, 3])

    problem = AmplificationProblem(
        oracle=oracle_circuit,
        is_good_state=lambda bitstr: bitstr == target_bitstr
    )

    backend = Aer.get_backend(backend_name)
    sampler = AerSampler(run_options={"shots": shots})

    best_counts = {}
    best_prob = 0.0
    best_iter = 0
    target_probabilities = []

    for k in range(1, max_iter + 1):
        grover = Grover(sampler=sampler, iterations=k)
        result = grover.amplify(problem)
        counts = result.circuit_results[0]
        total_counts = sum(counts.values())
        current_best_state = max(counts, key=counts.get)
        current_prob = counts[current_best_state] / total_counts
        target_prob = counts.get(target_bitstr, 0) / total_counts
        target_probabilities.append(target_prob)

        print(f"Iteration {k}: Highest = {current_best_state} (P = {current_prob:.3f}), "
              f"Target {target_bitstr} P = {target_prob:.3f}")

        if target_prob > best_prob:
            best_prob = target_prob
            best_counts = counts
            best_iter = k

        if best_prob >= threshold:
            print("Threshold probability reached. Early termination.")
            break

    circuit = grover.construct_circuit(problem)
    optimized_circuit = transpile(circuit, backend, optimization_level=optimization_level)
    print("\nOptimized Grover Circuit:")
    print(optimized_circuit.draw())

    return best_counts, best_iter, optimized_circuit, target_probabilities


def plot_results(experimental_probs, theoretical=True):
    max_iter = len(experimental_probs)
    iterations = np.arange(1, max_iter + 1)

    if theoretical:
        theta = np.arcsin(1 / np.sqrt(16))  # N = 2^4
        theory_probs = np.sin((2 * iterations + 1) * theta) ** 2
        plt.plot(iterations, theory_probs, color='gray', linestyle='--', marker='x', label="Theoretical", linewidth=1.5)

    plt.plot(iterations, experimental_probs, color='orange', marker='o', linewidth=2.5, label="Experimental")
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("Success Probability", fontsize=16)
    plt.xticks(iterations)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    target_state = "0110"  # ✅ 4-bit 目标态
    max_iter = 3
    threshold = 0.96132
    shots = 1024
    optimization_level = 3
    backend_name = 'qasm_simulator'

    best_counts, best_iter, optimized_circuit, target_probs = adaptive_grover_search_manual_oracle(
        target_state=target_state,
        max_iter=max_iter,
        threshold=threshold,
        shots=shots,
        optimization_level=optimization_level,
        backend_name=backend_name
    )

    print("\nFinal measurement results from Grover search:", best_counts)

    plot_distribution(best_counts)
    plt.title(f"Grover Search Result (Best Iteration: {best_iter})")
    plt.xlabel("Measured Bitstrings")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.show()

    plot_results(experimental_probs=target_probs)
# 保存为 PDF
    circuit_drawer(optimized_circuit, output='mpl', filename="grover_4bit_circuit.pdf", style='iqp')

if __name__ == '__main__':
    main()
