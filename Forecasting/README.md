# Global Temperature Anomaly Analysis 🌍📈

This project explores global temperature anomalies using the NOAA Global Surface Temperature Dataset. The goal is to analyze historical trends and patterns, forecast future anomalies, and compare methods for time series analysis.

## About the Dataset

The NOAA Global Surface Temperature Dataset (NOAAGlobalTemp) contains monthly global surface temperature anomalies from 1850 to the present. The dataset includes data from:
	•	Land temperatures (Global Historical Climatology Network - Monthly)
	•	Ocean temperatures (Extended Reconstructed Sea Surface Temperature)

Key information:
	•	Resolution: Monthly
	•	Timeframe: 1850–present
	•	Uses: Climate monitoring and trend analysis

## Project Objectives
	1.	Explore Temperature Trends: Identify long-term patterns in temperature anomalies.
	2.	Forecast Future Anomalies: Use models to predict temperature changes.
	3.	Compare Models:
	•	Neural Networks for continuous predictions.
	•	Hidden Markov Models for state-based pattern recognition.

## Methods

**Data Preparation**
	•	Data was cleaned and preprocessed for analysis.
	•	Monthly anomalies were scaled for better model performance.

**Exploratory Data Analysis (EDA)**
	•	Visualized temperature trends over time.
	•	Analyzed seasonal and long-term patterns.

## Models
	1.	**Feedforward Neural Network (DNN)**:
	•	Predicted next month’s temperature anomaly.
	•	Key layers: Input → 128 neurons → 64 neurons → Output.
	•	Evaluated using Mean Squared Error (MSE).
	2.	**Hidden Markov Model (HMM)**:
	•	Classified anomalies into 3 hidden states.
	•	Visualized state transitions and patterns.

## Evaluation
	•	Compared model outputs and analyzed performance.
	•	Metrics used: Prediction error (DNN) and transition clarity (HMM).

## Visualizations
	•	Temperature Trends: Graphs showing global warming patterns.
	•	Model Outputs:
	•	Neural Network predictions vs. actual values.
	•	HMM state classifications and transitions.

## Key Takeaways
	•	Temperature anomalies show clear upward trends.
	•	Neural Networks provide accurate short-term predictions.
	•	Hidden Markov Models are better at revealing distinct states in temperature patterns.
