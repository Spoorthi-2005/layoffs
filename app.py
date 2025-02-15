from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and encoders
with open("layoff_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("company_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load dataset
df = pd.read_excel(r"C:\Users\spoor\Downloads\layoffs\tech_layoffs.xlsx")

@app.route('/')
def home():
    companies = df["Company"].unique().tolist()
    industries = df["Industry"].unique().tolist()  # Ensure 'Industry' column exists
    return render_template("index.html", companies=companies, industries=industries)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    company_name = data.get("company")
    industry = data.get("industry")
    year = int(data.get("year"))

    if company_name not in label_encoder.classes_:
        return jsonify({"error": "Company not found in dataset"}), 400

    company_encoded = label_encoder.transform([company_name])[0]

    # Fetch previous layoffs
    company_data = df[(df["Company"] == company_name) & (df["Industry"] == industry)]
    
    if company_data.empty:
        return jsonify({"error": "No layoff data available for this company and industry"}), 400

    last_record = company_data.iloc[-1]  
    company_size_before = last_record["Company_Size_before_Layoffs"]
    company_size_after = last_record["Company_Size_after_layoffs"]
    percentage = last_record["Percentage"]

    input_data = np.array([[year, company_encoded, company_size_before, company_size_after, percentage]])
    predicted_layoffs = model.predict(input_data)[0]

    return jsonify({
        "predicted_layoffs": int(predicted_layoffs),
        "percentage": round(percentage, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
