import torch, pickle, argparse, pdb
from model.iimodel import IIMODEL


def train_single_step_model(
        exp_id,
        data_filename,
        dropout_ratio,
        obj_use_mean_return,
        steps,
        lr=0.0001,
        device='cuda',
        log_interval = 50,
        eval_interval = 50,
        seed = 5,
    ):
    
    data_id = data_filename.split('single_step')[1].split('alpaca')[0]
    model_id = 'single_action_m_'+ exp_id +'_dropout'+str(dropout_ratio)+'_objmeanret'+str(obj_use_mean_return)+'_steps'+str(steps)+'_lr'+str(lr)+'_'
    root_dir = '/home/ubuntu/code/HCL/invest/data/'+exp_id+'/'

    ## -- model training log -- 
    log_D = dict()
    log_D['data_filename'] = data_filename
    log_D['dropout_ratio'] = dropout_ratio
    log_D['steps'] = steps
    log_D['lr'] = lr
    log_D['log_interval'] = log_interval
    log_D['eval_interval'] = eval_interval
    log_D['seed'] = seed
    
    log_D['portfolio_shares'] = []
    log_D['actual_return'] = []
    log_D['mean_return'] = []
    log_D['stddev'] = []
    log_D['sharpe'] = []

    log_D['eval_portfolio_shares'] = []
    log_D['eval_actual_return'] = []
    log_D['eval_mean_return'] = []
    log_D['eval_stddev'] = []
    log_D['eval_sharpe'] = []

    log_D['top20_stocks'] = []
    
    torch.manual_seed(seed)
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your installation.")
        device = torch.device("cuda")
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available. Please check your installation.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu") 

    data = pickle.load(open(data_filename, 'rb'))
    features = data['trainFeature'].to(device)
    series = data['train_in_portfolio_series'].to(device)
    eval_features = data['testFeature'].to(device)
    eval_series = data['test_in_portfolio_series'].to(device)
    model = IIMODEL(dropout_ratio=dropout_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    features, series = features.to(device), series.to(device)
    eval_features, eval_series = eval_features.to(device), eval_series.to(device)
    
    for step in range(1, steps + 1):
        model.train()
        output = model(features)

        # assume we have a 1 dollar portfolio 
        # this is how many shares in the portfolio
        portfolio_shares = output / torch.unsqueeze((series[:, 0] + 1e-10), 1)
        actual_return = torch.sum(torch.unsqueeze((series[:, -1] - series[:, 0]), 1) * portfolio_shares)

        returns_series = torch.sum(series[:, 1:] * portfolio_shares - torch.unsqueeze(series[:, 0], 1) * portfolio_shares, dim=0)
        mean_return = torch.mean(returns_series, dim=0)
        stddev = torch.std(returns_series, dim=0)

        if obj_use_mean_return: 
            sharpe = mean_return / (stddev + 1e-10)
        else: 
            sharpe = actual_return / (stddev + 1e-10)
        loss = - sharpe 
        
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            log_D['portfolio_shares'].append(portfolio_shares)
            log_D['actual_return'].append(actual_return.item())
            log_D['mean_return'].append(mean_return.item())
            log_D['stddev'].append(stddev.item()) 
            log_D['sharpe'].append(sharpe.item())
            
            print('Train step: {} [{}/{} ({:.0f}%)]\tLoss (Sharpe ratio): {:.6f}\tMean Return: {:.3f}\tActual Return: {:.3f}\tStd Dev: {:.6f}'.format(
                step, 
                step, 
                steps,
                100. * step / steps, 
                loss.item(), 
                mean_return.item(),
                actual_return.item(),
                stddev.item(),
            ))
        
        if step % eval_interval == 0:
            model.eval()
            eval_output = model(eval_features)

            # assume we have a 1 dollar portfolio 
            # this is how many shares in the portfolio
            eval_portfolio_shares = eval_output / torch.unsqueeze((eval_series[:, 0] + 1e-10), 1)
            eval_actual_return = torch.sum(torch.unsqueeze((eval_series[:, -1] - eval_series[:, 0]), 1) * eval_portfolio_shares)

            eval_returns_series = torch.sum(eval_series[:, 1:] * eval_portfolio_shares - torch.unsqueeze(eval_series[:, 0], 1) * eval_portfolio_shares, dim=0)

            eval_mean_return = torch.mean(eval_returns_series, dim=0)
            eval_stddev = torch.std(eval_returns_series, dim=0)
            
            #eval_sharpe = eval_mean_return / eval_stddev 
            eval_sharpe = eval_actual_return / (eval_stddev + 1e-10)
            eval_loss = - eval_sharpe 

            _, top20_stocks_indices = torch.topk(eval_output, 20, dim=0)
            top20_stocks = []
            for i in range(len(top20_stocks_indices)):
                top20_stocks.append(data['all_test_tickers'][top20_stocks_indices[i]])
            print(f'--> Eval model:\tLoss (Sharpe ratio): {str(eval_loss.item())}\tMean Returns:{str(eval_mean_return)}\t Actual Returns: {str(eval_actual_return.item())}\tStd Dev: {str(eval_stddev.item())}') 
            print(f'--> Top 20 stocks: {str(top20_stocks)}')
            
            model_pt_filename = root_dir + model_id+'_' + data_id + '_step'+str(step)+'.pt'
            torch.save(model.state_dict(), model_pt_filename)

            log_D['eval_portfolio_shares'].append(eval_portfolio_shares)
            log_D['eval_actual_return'].append(eval_actual_return.item())
            log_D['eval_mean_return'].append(eval_mean_return.item())
            log_D['eval_stddev'].append(eval_stddev.item())
            log_D['eval_sharpe'].append(eval_sharpe.item())
            log_D['top20_stocks'].append(top20_stocks)
        
        model_log_filename = root_dir + model_id + '_'+data_id + '_log.pkl'
        pickle.dump(log_D, open(model_log_filename, 'wb'))
