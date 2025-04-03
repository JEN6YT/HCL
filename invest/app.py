from flask import Flask, jsonify
from update_prod import update_price_history_data, update_train_test_model_4d, update_train_test_model_25d

app = Flask(__name__)

@app.route('/api/model', methods=['GET'])
def get_portfolio():
    data = {
        'tickers':[
            'AAPL',
            'MSFT',
            'NVDA',
        ],
        'scores':[
            0.3,
            0.6,
            0.1,
        ],
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
