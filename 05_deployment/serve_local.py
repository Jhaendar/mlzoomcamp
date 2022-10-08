from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


# load dv
def load_dv(fname):
    with open(fname, 'rb') as file:
        dv = pickle.load(file)
    return dv
    
# load model
def load_model(fname):
    with open(fname, 'rb') as file:
        model = pickle.load(file)
    return model

# predict 
def predict_one(dv, model, data):
    X = dv.transform([data])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

if __name__ == "__main__":
    client_data = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    dv = load_dv('dv.bin')
    model = load_model('model1.bin')
    
    y_pred = predict_one(dv, model, client_data)
    print(y_pred)


