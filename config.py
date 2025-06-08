#from CFRNet.utils import get_computing_device
# from utils import get_computing_device
import torch.cuda


class Config:
    def __init__(self):
        self.num_epochs = 15 # MSE:70 ZILN:

        # Learning rate 1e-4 seems to be better with CfrNet
        # 1e-3 seems to be better with LogisticRegressionNet
        self.learning_rate = 1e-4

        # The rate of the weight decay was not mentioned in the paper.
        self.weight_decay = 1e-5  # Regularization term in pytorch
         
        self.batch_size = 6400 
        self.split_h = True
        self.ipm_function = 'wasserstein'  # Use mmd or wasserstein
        self.alpha = 1
        self.dataset = "histrom_binary_men"  # Either jobs or ihdp or kuaishou
        self.do_save = True  # Whether to save the output
        self.do_log_epochs = True  # Whether to log the number of epochs

        # Size of hidden dimensions
        self.hidden_dim_rep = 200
        self.hidden_dim_hypo = 100
        self.prefer_gpu = True  # Set to True if you want to use the GPU
        self.use_gpu = self.prefer_gpu and torch.cuda.is_available()
        # use gpu if available, otherwise use cpu, don't have get_computing_device function
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Model to use
        self.model_name = "cfrnet"  # Either logistic, cfrnet, or tarnet。

        self.output_dir = "output"
        self.save_normalized_model = False
        self.save_main_model = True
        self.random_seed = 40 # CFR_mmd: Men: 40, Wom:35 / CFR_wass: Men:10, Wom:20
