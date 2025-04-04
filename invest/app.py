import pickle
from flask import Flask, jsonify

app = Flask(__name__)

class StartService:
    def __init__(self):
        self.D4d = pickle.load(open('/home/ubuntu/code/HCL/invest/data/prod/prod_4d_model_prediction.pkl', 'rb'))
        self.D25d = pickle.load(open('/home/ubuntu/code/HCL/invest/data/prod/prod_25d_model_prediction.pkl', 'rb'))

    def predict_4d(self): 
        return self.D4d
    def predict_25d(self): 
        return self.D25d

predictService = StartService() 

@app.route('/api/model4d', methods=['GET'])
def get_portfolio_4d():
    data = predictService.predict_4d()
    return jsonify(data)

@app.route('/api/model25d', methods=['GET'])
def get_portfolio_25d():
    data = predictService.predict_25d()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
