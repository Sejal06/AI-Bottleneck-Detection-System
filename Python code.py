import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Step 1: Generate Production Data
# -----------------------------

np.random.seed(42)

data = {
    "Station_A": np.random.normal(50,5,100),
    "Station_B": np.random.normal(48,4,100),
    "Station_C": np.random.normal(90,8,100),
    "Station_D": np.random.normal(55,5,100)
}

df = pd.DataFrame(data)

print("\nSample Production Data\n")
print(df.head())


# -----------------------------
# Step 2: Calculate Average Cycle Time
# -----------------------------

avg_cycle = df.mean()

print("\nAverage Cycle Time (seconds):\n")
print(avg_cycle)


# -----------------------------
# Step 3: Detect Bottleneck
# -----------------------------

bottleneck = avg_cycle.idxmax()

print("\nDetected Bottleneck Station:", bottleneck)


# -----------------------------
# Step 4: Visualize Cycle Time
# -----------------------------

plt.figure()

avg_cycle.plot(kind="bar")

plt.title("Average Cycle Time by Station")
plt.xlabel("Stations")
plt.ylabel("Cycle Time (seconds)")

plt.show()


# -----------------------------
# Step 5: Calculate Line Throughput
# -----------------------------

line_cycle_time = avg_cycle.max()

throughput = 3600 / line_cycle_time

print("\nLine Cycle Time:", line_cycle_time)

print("Estimated Throughput per hour:", throughput)


# -----------------------------
# Step 6: Cycle Time Trend Analysis
# -----------------------------

plt.figure()

plt.plot(df["Station_C"])

plt.title("Cycle Time Trend - Station C")
plt.xlabel("Production Cycle")
plt.ylabel("Cycle Time (seconds)")

plt.show()


# -----------------------------
# Step 7: Predict Future Cycle Time
# -----------------------------

X = np.arange(len(df)).reshape(-1,1)

y = df["Station_C"]

model = LinearRegression()

model.fit(X,y)

future = np.array([[120]])

prediction = model.predict(future)

print("\nPredicted Future Cycle Time:", prediction)


# -----------------------------
# Step 8: AI Recommendation
# -----------------------------

if line_cycle_time > 80:
    print("\nAI Recommendation:")
    print("• Improve tooling at bottleneck station")
    print("• Add parallel workstation")
    print("• Rebalance production line")
