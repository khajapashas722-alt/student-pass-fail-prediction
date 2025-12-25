# 🎓 Student Pass/Fail Prediction System

This project predicts whether a student will **PASS or FAIL** using Machine Learning.

## 📌 Project Overview
The system analyzes student academic and behavioral data to classify outcomes as Pass or Fail.  
An interactive web application is built using **Streamlit** for easy user interaction.

## 🧠 Machine Learning Model
- Logistic Regression
- Train-Test Split
- Feature Scaling using StandardScaler

## 📊 Features Used
- Study Time
- Number of Absences
- G1 (First Period Grade)
- G2 (Second Period Grade)

A student is considered to **PASS** if the final grade (G3) is ≥ 10.

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## 📂 Project Structure
```
app.py # Streamlit application
student-mat.csv # Dataset
student-por.csv # Dataset
student-merge.R # Data preprocessing script
requirements.txt # Project dependencies
```


## ▶️ How to Run the Project
1. Install required libraries:
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app
```bash
streamlit run app.py
```
## 👨‍💻 Author
Shaik Khaja Pasha

