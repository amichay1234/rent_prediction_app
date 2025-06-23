from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from assets_data_prep import prepare_data

app = Flask(__name__)

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {key: request.form.get(key) for key in request.form.keys()}
        print("📥 Raw form data:", input_data)

        for key, value in input_data.items():
            if value is None or value == '':
                input_data[key] = np.nan
            elif isinstance(value, str) and value.lower() in ['true', 'on', 'yes']:
                input_data[key] = 1
            elif isinstance(value, str) and value.lower() in ['false', 'off', 'no']:
                input_data[key] = 0
            else:
                try:
                    input_data[key] = float(value)
                except:
                    input_data[key] = value

        input_df = pd.DataFrame([input_data])

        if 'property_type' not in input_df.columns:
            input_df['property_type'] = np.nan
        if 'description' not in input_df.columns:
            input_df['description'] = ""
        if 'price' not in input_df.columns:
            input_df['price'] = np.nan

        input_df['description'] = input_df['description'].fillna("").astype(str)

        boolean_fields = [
            'has_parking', 'has_storage', 'elevator', 'ac', 'handicap',
            'has_bars', 'has_safe_room', 'has_balcony', 'is_furnished', 'is_renovated'
        ]
        for field in boolean_fields:
            if field not in input_df.columns:
                input_df[field] = 0

        print("✅ DataFrame לפני prepare_data:", input_df.columns.tolist())

        # ✅ קריאה לפונקציה במצב תחזית
        processed_df = prepare_data(input_df, is_prediction=True)

        print("✅ DataFrame אחרי prepare_data:", processed_df.columns.tolist())

        predicted_price = model.predict(processed_df.drop(columns=["price"], errors='ignore'))[0]
        predicted_price = round(predicted_price, 2)

        return render_template("index.html", prediction=predicted_price)

    except Exception as e:
        print("❌ שגיאת חיזוי:", str(e))
        return render_template("index.html", prediction=f"שגיאה: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
