from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load files
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    gender = request.form['gender']
    senior = int(request.form['SeniorCitizen'])
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    tenure = int(request.form['tenure'])

    phoneservice = request.form['PhoneService']
    multiplelines = request.form['MultipleLines']
    internetservice = request.form['InternetService']

    onlinesecurity = request.form['OnlineSecurity']
    onlinebackup = request.form['OnlineBackup']
    deviceprotection = request.form['DeviceProtection']
    techsupport = request.form['TechSupport']

    streamingtv = request.form['StreamingTV']
    streamingmovies = request.form['StreamingMovies']

    contract = request.form['Contract']
    paperlessbilling = request.form['PaperlessBilling']
    paymentmethod = request.form['PaymentMethod']

    monthlycharges = float(request.form['MonthlyCharges'])
    totalcharges = float(request.form['TotalCharges'])

    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': multiplelines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': paymentmethod,
        'MonthlyCharges': monthlycharges,
        'TotalCharges': totalcharges
    }

    df = pd.DataFrame([data])

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability>=0.75:
        risk = "High Risk"
        suggestion = "Offer discount and retention plan "
    elif probability>=0.50:
        risk = "Medium Risk"
        suggestion = "Provide loyalty benefits"
    else:
        risk = "Low Risk"
        suggestion = "Customer is Stable"

    result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability*100,2),
        risk=risk,
        suggestion=suggestion
    )

   

if __name__ == "__main__":
    app.run(debug=True)