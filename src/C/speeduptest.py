import subprocess
import pandas as pd
import os
import matplotlib.pyplot as plt

schemes = ["ngp", "cic", "tsc"]
threads = [1, 2, 4, 8]  # Adjust as needed for your system
csv_file = "timing_results.csv"

# Initialize CSV with header
with open(csv_file, "w") as f:
    f.write("dp,threads,avg_time\n")

# Run benchmark and collect data
records = []

for dp in schemes:
    for t in threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t)
        env["PM_DP"] = dp  # optional: allow passing dp to your C++ code if needed

        print(f"Running: dp={dp}, threads={t} ...")
        result = subprocess.run(["./sim"], capture_output=True, env=env, text=True)

        avg_time = None
        for line in result.stdout.splitlines():
            if "Average per step" in line:
                # Expect line like: Average per step: 0.012345 seconds
                try:
                    avg_time = float(line.strip().split(":")[1].split()[0])
                    records.append({
                        "dp": dp,
                        "threads": t,
                        "avg_time": avg_time
                    })
                    break
                except Exception as e:
                    print("Error parsing:", line)

# Save to CSV
df = pd.DataFrame(records)
df.to_csv(csv_file, index=False)
print("Results written to", csv_file)

# Plot speedup
plt.figure()
for dp in df["dp"].unique():
    subset = df[df["dp"] == dp].sort_values("threads")
    time1 = subset[subset["threads"] == 1]["avg_time"].values[0]
    speedup = time1 / subset["avg_time"]
    plt.plot(subset["threads"], speedup, marker='o', label=dp.upper())

plt.xlabel("Threads")
plt.ylabel("Speedup (vs 1 thread)")
plt.title("Parallel Speedup of PM Integrator")
plt.legend(title="Deposition")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_plot.png")
plt.show()