import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm
import random


''' Create a new file in which you will now train a Hidden Markov Model for the same task. Copy your code for problems
1.1 to 1.5. No additional points for that. Make sure to import hmmlearn.'''

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

# 15. Train a Gaussian Hidden Markov Model to fit the data. Parameters (number of hidden states and
# number of iterations are your choice; choose for example 3 states and 100 iterations).
model = hmm.GaussianHMM(n_components=3, n_iter=100, random_state=42)
model.fit(anomaly_array)

# 16. Predict the hidden states
predict = model.predict(anomaly_array)
print(predict)

# 17. (4 points) Create a plot that colors every temperature anomaly (every month) in the color of its corresponding hidden state.
# Background: the model assigned a hidden state to every anomaly with the .predict() command.
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

# 18. (4 points) Visualize the state transition matrix using sns.heatmap.
# You can access the probabilities using model.transmat_.
plt.figure()
sns.heatmap(model.transmat_, annot=True, cmap='viridis', fmt='.3f')
plt.title("State Transition Matrix Heatmap")
plt.xlabel("To State")
plt.ylabel("From State")
plt.tight_layout()
plt.show()


