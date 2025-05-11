import matplotlib.pyplot as plt
from main import run_path_planning

times = []
num_runs = 20
for i in range(num_runs):
    t = run_path_planning()
    times.append(t)
    print(f"Run {i+1}: {t:.4f} seconds")

plt.figure(figsize=(8,5))
plt.plot(range(1, num_runs+1), times, marker='o', color='dodgerblue')
plt.xlabel('No of Runs', fontsize=14)
plt.ylabel('Time in Sec', fontsize=14)
plt.title('Computation Time per Run', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('timing_comparison.png')
# plt.show()  # Do not display the figure 