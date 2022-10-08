from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
import pickle

# HELPER METHODS 
# load dv
def load_dv(fname):
    """Loads the DictVectorizer from file

    Args:
        fname (str): Filename

    Returns:
        DictVectorizer: DictVectorizer
    """ 
    with open(fname, 'rb') as file:
        dv = pickle.load(file)
    return dv
    
# load model
def load_model(fname):
    """Loads the Model from file

    Args:
        fname (string): filename of the model

    Returns:
        Model: Model
    """
    with open(fname, 'rb') as file:
        model = pickle.load(file)
    return model

# predict 
def predict_one(dv, model, data):
    """Single predict using supplied model and dv

    Args:
        dv (DictVectorizer): dv
        model (Model): Model
        data (dict): Single customer record

    Returns:
        float: probability
    """
    X = dv.transform([data])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


dv = load_dv('dv.bin')
model = load_model('model1.bin')

app = Flask('flask-cc-app')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    pred = predict_one(dv, model, data)
    
    result = {
        'prob': float(pred)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)