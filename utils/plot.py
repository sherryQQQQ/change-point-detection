import matplotlib.pyplot as plt
import os


def plot_pmf(p, Z_piece, mu, m, t, N, file,save_path):
    plt.figure(figsize=(10, 6))
    plt.scate(range(len(p)), p, width=0.8, alpha=0.7)
    plt.title(f'PMF at t={t} (N={N} ), file: {file}')
    plt.xlabel('Queue Length')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)

    filename = f"pmf_plot_t{t}_{os.path.splitext(file)[0]}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")
