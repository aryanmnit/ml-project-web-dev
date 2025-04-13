import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
ml_model = pickle.load(open("ml_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')  # Render home page first

@app.route('/quiz')
def quiz():
    return render_template('quiz.html', result='')  # Load quiz 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        answers = [int(request.form[f'q{i}csv']) for i in range(1, 20)]
        arr_reshaped = np.array(answers).reshape(1, -1)
        result = ml_model.predict(arr_reshaped)[0]
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)