from flask import Flask, render_template,request
import pickle
import pandas as pd

app = Flask(__name__)

with open("./notebooks/model.pkl","rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model_page():
    return render_template("model.html")

@app.route("/predict",methods=["POST"])
def predict():
    amount = float(request.form["amount"])
    country = request.form["country"]
    transaction_type = request.form["transaction_type"]
    merchant_category = request.form["merchant_category"]
    
    input_data = pd.DataFrame({
        "transaction_id": [0],
        "user_id": [0],
        "amount": [amount],
        "transaction_type": [transaction_type],
        "merchant_category": [merchant_category],
        "country": [country],
        "hour": [1],
        "device_risk_score": [0],
        "ip_risk_score": [0]
    })
    
    prediction = model.predict(input_data)[0]

    return render_template("result.html",
                          prediction=prediction,
                          amount=amount,
                          country=country,
                          transaction_type=transaction_type,
                          merchant_category=merchant_category)
    
if __name__ == '__main__':
    app.run(debug=False)
    
    
