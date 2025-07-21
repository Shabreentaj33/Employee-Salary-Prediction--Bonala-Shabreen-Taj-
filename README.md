
# 💼 Employee Salary Prediction

This project predicts whether an employee earns more than $50K per year based on demographic and work-related features.  
It uses a machine learning model trained on the UCI Adult Income dataset and is deployed using Streamlit.

## 🔧 Features
- Input employee attributes via an interactive web interface
- Predict salary category: `>50K` or `<=50K`
- Batch prediction support
- Deployed using Streamlit + ngrok (Colab compatible)

## 📦 Requirements
Install required packages using:

```bash
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
streamlit run streamlit_app.py
```

## 🧠 Model
The model was trained using scikit-learn and saved as `best_model.pkl`.
