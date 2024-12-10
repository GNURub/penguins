import pickle
import numpy as np
from flask import Flask, jsonify, request

classes = ['Adelie', 'Chinstrap', 'Gentoo']

def predict_single(penguin, vectorizer, scaler, model):
    # No se como hacer participe a las variables categoricas para la prediccion
    penguin_cat = vectorizer.transform([{
         'island': penguin['island'],
         'sex': penguin['sex']
     }])
    
    penguin_num = scaler.transform([[
        penguin['bill_length_mm'],
        penguin['bill_depth_mm'],
        penguin['flipper_length_mm'],
        penguin['body_mass_g']
    ]])

    penguin_val = np.hstack((penguin_cat, penguin_num))
    
    y_pred = model.predict(penguin_val)[0]
    y_prob = model.predict_proba(penguin_val)[0][y_pred]
    return (y_pred, y_prob)

def predict(vectorizer, scaler, model):
    penguin = request.get_json()
    especie, probabilidad = predict_single(penguin, vectorizer, scaler, model)
    result = {
        'especie': classes[especie],
        'probabilidad': float(probabilidad)
    }
    return jsonify(result)

app = Flask('penguins')

@app.route('/predict/<alg>', methods=['POST'])
def predict_type(alg):
    model_files = {
        'lr': 'models/lr.pck',
        'svm': 'models/svm.pck',
        'dt': 'models/dt.pck',
        'knn': 'models/knn.pck'
    }

    if alg not in model_files:
        return jsonify({'error': 'not valid type'}), 400

    try:
        with open(model_files[alg], 'rb') as f:
            vectorizer, scaler, model = pickle.load(f)
    except:
        return jsonify({'error': f'Model file not found for type: {alg}'}), 500

    return predict(vectorizer, scaler, model)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
