import os
import joblib
import smtplib
import numpy as np
from flask import Flask, render_template, request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from xhtml2pdf import pisa

app = Flask(__name__)
MODEL_DIR = "models"

# Load models and encoder
species_encoder = joblib.load(os.path.join(MODEL_DIR, "species_encoder.pkl"))
microbes = list(species_encoder.classes_)
antibiotics = [f.replace("model_", "").replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.startswith("model_")]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    alternatives = None

    if request.method == "POST":
        selected_microbe = request.form.get("microbe")
        selected_antibiotic = request.form.get("antibiotic")
        patient_email = request.form.get("email")

        try:
            microbe_encoded = species_encoder.transform([selected_microbe])[0]
            model_path = os.path.join(MODEL_DIR, f"model_{selected_antibiotic}.pkl")
            model = joblib.load(model_path)

            prediction = model.predict(np.array([[microbe_encoded]]))[0]
            probability = model.predict_proba(np.array([[microbe_encoded]]))[0][int(prediction)] * 100

            if prediction == 1:
                prediction_result = f"The microbe {selected_microbe} is Resistant to {selected_antibiotic} ({probability:.2f}%)."
                with open(os.path.join(MODEL_DIR, "alternative_antibiotics.json")) as f:
                    import json
                    alt_map = json.load(f)
                alternatives = alt_map.get(selected_microbe, [])
            else:
                prediction_result = f"The microbe {selected_microbe} is Susceptible to {selected_antibiotic} ({probability:.2f}%)."

            pdf_path = generate_pdf(selected_microbe, selected_antibiotic, prediction_result, alternatives)
            send_email(patient_email, prediction_result, pdf_path)

        except Exception as e:
            prediction_result = f"Error: {e}"

    return render_template("index.html", microbes=microbes, antibiotics=antibiotics, prediction=prediction_result, alternatives=alternatives)

def generate_pdf(microbe, antibiotic, result, alternatives):
    html = render_template("report_template.html", microbe=microbe, antibiotic=antibiotic, result=result, alternatives=alternatives)
    filename = f"report_{microbe}_{antibiotic}.pdf"
    pdf_path = os.path.join("generated_reports", filename)
    os.makedirs("generated_reports", exist_ok=True)
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(html, dest=f)
    return pdf_path

def send_email(to_email, message, attachment_path):
    sender = "[Replace with your mail id]"
    password = "[Replace with your secret passkey]"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = "Antibiotic Resistance Prediction Result"

    msg.attach(MIMEText(message, "plain"))

    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attachment_path)}"'
        msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)

if __name__ == "__main__":
    app.run(debug=True)
