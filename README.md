# 🕒 Appliances Energy Prediction

A machine learning project developed for the **Python for Machine Learning and Data Science** course. The goal was to create a model capable of predicting the **time of day** based on household environmental conditions and energy usage patterns.

---

## 📊 Project Overview

The model predicts **Seconds From Midnight (SFM)** based on 28 continuous features such as temperature, humidity, and power consumption data. The dataset includes over 19,000 samples collected at 10-minute intervals over a span of 4.5 months.

### 🔍 Hypothesis
> It is feasible to predict the time of day with an error margin of less than 30 minutes (1800 seconds) using the provided environmental and energy data.

---

## 📁 Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- Collected via:
  - ZigBee wireless sensors (indoor temp/humidity)
  - m-bus energy meters (appliances & lights)
  - Weather station data from Chievres Airport, Belgium

---

## 📈 Correlation Matrix

A Pearson correlation matrix was generated to identify dependencies between features and the target variable (`SFM`).

Key findings:
- `T2` (Living Room Temperature), `RH_out` (Outdoor Humidity), and `Lights` showed the strongest correlations with `SFM`
- These features were used for initial models
- Grouped clusters of temperature and humidity features are clearly visible

![image](https://github.com/user-attachments/assets/ce0b88a0-62c3-4825-b0ad-86297234403b)


---

## 🔬 Feature Engineering

- Time column was converted into `Seconds From Midnight (SFM)`
- Feature selection methods:
  - Pearson Correlation
  - Principal Component Analysis (PCA)
  - Full feature set (“Basic Set”)
- Features were scaled using `StandardScaler`
- Tuned hyperparameters and tested hidden layer sizes for MLP

---

## ⚙️ Models Used

Machine learning models implemented:

| Model                  | Best RMSE (s)       | Feature Set          |
|------------------------|---------------------|----------------------|
| Polynomial Regression  | ~20,957             | Selected Features    |
| Linear Regression      | ~18,415             | Basic Set            |
| Support Vector Machine | ~20,677             | PCA                  |
| **MLP (Final Model)**  | **~10,022**         | Basic Set            |

---

## 🚧 Conclusions

- None of the models achieved the target RMSE of ≤1800 seconds
- MLP gave the most promising results but still had high prediction error (~2h45min)
- Likely issues:
  - Limited seasonal data range
  - Lack of strong, consistent temporal signals in features
  - Complexity and nonlinearity of the task

