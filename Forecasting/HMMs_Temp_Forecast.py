import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm
import random

'''This is part of a project where I am testing out a DNN forecasting model against the Hidden Markov Model'''

# read the file
df = pd.read_csv("temperature.csv")

# convert the year column into datetime
df['Year'] = pd.to_datetime(df['Year'])

# Plot anomaly vs year
plt.figure()
plt.plot(df['Year'], df['Anomaly'])
plt.xlabel('Year')
plt.ylabel('Temp Anomaly')
plt.show()

# Save only the temperature anomaly into a 2D numpy array
anomaly_array = df['Anomaly'].values.reshape(-1, 1)

# Training a Gaussian Hidden Markov Model to fit the data. Parameters (number of hidden states and
# number of iterations are your choice; choose for example 3 states and 100 iterations).
model = hmm.GaussianHMM(n_components=3, n_iter=100, random_state=42)
model.fit(anomaly_array)

# Predicting the hidden states
predict = model.predict(anomaly_array)
print(predict)

# Creating a plot that colors every temperature anomaly (every month) in the color of its corresponding hidden state.
states = pd.unique(predict)
print("Unique states:", states)

plt.figure(figsize=(15, 10))
for i in states:
    plt.plot(df["Year"][predict == i], df["Anomaly"][predict == i], '.', label=f'State {i}')
plt.legend()
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.title("Temperature Anomalies Categorized by Hidden States")
plt.grid(True)
plt.show()

# Visualizing the state transition matrix using sns.heatmap.
plt.figure()
sns.heatmap(model.transmat_, annot=True, cmap='viridis', fmt='.3f')
plt.title("State Transition Matrix Heatmap")
plt.xlabel("To State")
plt.ylabel("From State")
plt.tight_layout()
plt.show()


