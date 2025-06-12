import subprocess
import pandas as pd
import os
import matplotlib.pyplot as plt

schemes = ["tsc", "cic", "ngp"]
threads = [1, 2, 4, 8]
csv_file = "timing_results.csv"

# Initialize CSV
with open(csv_file, "w") as f:
    f.write("dp,threads,time_taken\n")

records = []

for dp in schemes:
    for t in threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t)
        env["PM_DP"] = dp  # used if per.cpp reads it

        print(f"Running: dp={dp}, threads={t} ...")
        result = subprocess.run(["./per"], capture_output=True, env=env, text=True)

        avg_time = None
        for line in result.stdout.splitlines():
            if "Time taken" in line:
                try:
                    avg_time = float(line.strip().split(":")[1].split()[0])
                    records.append({
                        "dp": dp,
                        "threads": t,
                        "time_taken": avg_time
                    })
                    break
                except Exception as e:
                    print("Error parsing:", line)

# Save results
df = pd.DataFrame(records)
df.to_csv(csv_file, index=False)
print("Results written to", csv_file)

plt.figure()
for dp in df["dp"].unique():
    subset = df[df["dp"] == dp].sort_values("threads")
    plt.plot(subset["threads"], subset["time_taken"], marker='o', label=dp.upper())

plt.xlabel("Threads")
plt.ylabel("Time Taken for 200 steps (seconds)")
plt.title("Performance of PM Integrator vs Threads")
plt.legend(title="Deposition")
plt.grid(True)
plt.tight_layout()
plt.savefig("time_vs_threads.png")
plt.show()

plt.figure()
for dp in df["dp"].unique():
    subset = df[df["dp"] == dp].sort_values("threads")
    time1 = subset[subset["threads"] == 1]["time_taken"].values[0]
    speedup = time1 / subset["time_taken"]
    plt.plot(subset["threads"], speedup, marker='o', label=dp.upper())

plt.xlabel("Threads")
plt.ylabel("Speedup (vs 1 thread)")
plt.title("Parallel Speedup of PM Integrator (per.cpp)")
plt.legend(title="Deposition")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_plot.png")
plt.show()
