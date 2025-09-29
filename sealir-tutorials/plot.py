import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read data from output.txt and parse into plotting format
filename = 'output.txt'
d = []

with open(filename) as f:
    lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        fn = lines[i]
        a_times = [float(x) for x in lines[i+1][1:-1].split(',')]
        b_times = [float(x) for x in lines[i+2][1:-1].split(',')]
        for t in a_times:
            d.append({'function': fn, 'group': 'NumPy', 'time': t})
        for t in b_times:
            d.append({'function': fn, 'group': 'MLIRGen', 'time': t})
        # d.append({"function": fn, "a_times": a_times, "b_times": b_times})
        i += 3
import json
json.dumps(d)
import numpy as np
df = pd.DataFrame(d)
# Plot
plt.figure(figsize=(8, 16))
sns.boxplot(x='function', y='time', hue='group', data=df, fliersize=0)
plt.title('Function Time Comparison (NumPy vs MLIR)')
plt.ylabel('Time (Microseconds)')
plt.xlabel('Function Name')
plt.tick_params(axis='x', labelrotation=90)
positions = np.arange(len(df['function'].unique()))
for pos in positions:
    plt.axvline(x=pos, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('function_times_boxplot.png')
