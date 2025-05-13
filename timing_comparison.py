import matplotlib.pyplot as plt
import time
from main import main  # Import only the main() function

times = []
num_runs = 20

for i in range(num_runs):
    start_time = time.time()
    main()  # Run the full path planning logic
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_runs + 1), times, marker='o', color='dodgerblue')
plt.xlabel('No of Runs', fontsize=14)
plt.ylabel('Time in Sec', fontsize=14)
plt.title('Computation Time per Run', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('timing_comparison.png')
