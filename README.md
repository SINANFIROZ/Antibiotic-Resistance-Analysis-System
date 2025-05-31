🧬 Antibiotic Resistance Analysis System

📌 Project Description

This project aims to predict whether a microbe is resistant or susceptible to a given antibiotic using a machine learning model. By analyzing features like patient data and microbial attributes, the system outputs the probability (%) of resistance or susceptibility. Additionally, it suggests alternative antibiotics in case resistance is detected. This tool is designed to support healthcare providers in making more informed decisions and combating antimicrobial resistance.

🛠️ Technologies Used

- 🐍 Python (XGBoost, Pandas, Scikit-learn)
- 🌐 Flask (for backend API)
- 🖥️ HTML, CSS, JavaScript (for frontend)
- 📊 Machine Learning (XGBoost classifier)
- 📦 SQLite or CSV (for dataset handling)

🚀 How to Run

1. Clone the repository
bash
git clone https://github.com/your-username/antibiotic-resistance-prediction.git
cd antibiotic-resistance-prediction

2.Setup and activate virtual environment(Optional)
python -m venv venv
# Activate the environment:
# For Linux/Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

3.Run the flask backend
python app.py

NOTE: Make sure to install all dependencies


📊Features
-Predicts resistance/susceptibility with probability
-Suggests alternative antibiotics for resistant cases
-Easy-to-use web interface for entering input data
-Built-in XGBoost model for accurate predictions

📷Screenshots
![Screenshot 2025-05-06 150044](https://github.com/user-attachments/assets/93fe3f5d-0a51-400f-9654-47b0c74952b0)
![Screenshot 2025-05-06 150155](https://github.com/user-attachments/assets/d6a405ab-f072-4b15-959f-8ebfcbe1dcd7)



