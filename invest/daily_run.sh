
python3 update_prod.py > /home/ubuntu/code/HCL/invest/data/prod/latest_update_log 
#python3 trade.py > /home/ubuntu/code/HCL/invest/data/prod/latest_trade_log 
pkill gunicorn; gunicorn -b 0.0.0.0:8000 -w 2 app:app
