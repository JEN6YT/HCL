import torch, pickle, argparse, pdb
from model.iimodel import IIMODEL

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: -1 for full dataset)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--steps', type=int, default=300, metavar='N',
                        help='number of steps to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device for compute (default: cpu)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your installation.")
        device = torch.device("cuda")
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available. Please check your installation.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu") 

    """
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if device == torch.device("cuda"):
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    """
    data = pickle.load(open('data/model_data_single_step_v1.pkl', 'rb'))
    features = data['trainFeature']
    series = data['train_in_portfolio_series']
    model = IIMODEL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # testing 
    scores = model(features)
    print(scores)

    model.train()
    for step in range(1, args.steps + 1):
        features, series = features.to(device), series.to(device)
        optimizer.zero_grad()
        output = model(features)

        # assume we have a 1 dollar portfolio 
        # this is how many shares in the portfolio
        portfolio_shares = output / torch.unsqueeze((series[:, 0] + 1e-10), 1)
        actual_return = torch.sum(torch.unsqueeze((series[:, -1] - series[:, 0]), 1) * portfolio_shares)
        returns_series = torch.sum(series[:, 1:] * portfolio_shares - torch.unsqueeze(series[:, 0], 1) * portfolio_shares, dim=0)
        stddev = torch.std(returns_series, dim=0)
        
        sharpe = actual_return / stddev 
        loss = - sharpe 
        
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train step: {} [{}/{} ({:.0f}%)]\tLoss (Sharpe ratio): {:.6f}\tActual Return: {:.3f}\tStd Dev: {:.6f}'.format(
                step, 
                step, 
                args.steps,
                100. * step / args.steps, 
                loss.item(), 
                actual_return.item(),
                stddev.item(),
            ))
            if args.save_model:
                torch.save(model.state_dict(), f"ckpts/invest_single_step_model_test_step{str(step)}.pt")
            if args.dry_run:
                break
        #test(model, device, test_loader)


if __name__ == '__main__':
    main()