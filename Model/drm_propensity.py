import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np

class SimpleTCModelDNN_propensity(nn.Module):
    def __init__(self, 
                input_dim, 
                num_hidden,
                p_model_specs={'model': linear_model.LogisticRegression, 'params': {'C': 1.0, 'max_iter': 1000}}, 
                k_fold=5
    ):
        """
        Initializes the Direct Ranking Model.

        Args:
        - input_dim: int, dimensionality of user context features (both treatment and control).
        - num_hidden: int, number of hidden units. If 0, skips the hidden layer.
        """
        super(SimpleTCModelDNN_propensity, self).__init__()
        self.p_model_specs = p_model_specs
        self.p_model = None
        self.k_fold = k_fold

        self.num_hidden = num_hidden

        if num_hidden > 0:
            self.hidden_layer = nn.Linear(input_dim, num_hidden)
            self.ranker = nn.Linear(num_hidden, 1)
        else:
            self.ranker = nn.Linear(input_dim, 1)
    
    def fit_predict_p_hat(self, X, w):
        """
        Given X and T, fix a model to predict T (e) given X
        Fit and predict for p_hat.
        :param X: feature matrix
        :param w: binary indicator for treatment / control
        :return: a numpy array of predicted p_hat, same shape as w
        """
        if self.p_model_specs is None:
            return np.sum(w) / float(len(w)) * np.ones_like(w)
        
        kf = KFold(n_splits=self.k_fold)
        
        p_hat = np.zeros_like(w)
        
        # initialize m model 
        p_model = self.p_model_specs['model'](**self.p_model_specs['params']) 
        
        for fit_idx, pred_idx in kf.split(X):
            
            # split data into fit and predict
            X_fit, X_pred = X[fit_idx], X[pred_idx]
            w_fit = w[fit_idx]
                        
            p_model.fit(X_fit, w_fit)
            p_hat[pred_idx] = p_model.predict_proba(X=X_pred)[:,1]
        
        p_hat = np.clip(p_hat, 0 + 1e-7, 1 - 1e-7) 
        
        return p_hat
    
    def forward(self, D_tre, D_unt):
        """
        Forward pass for the model.

        Args:
        - D_tre: torch.Tensor, features for treatment group.
        - D_unt: torch.Tensor, features for control group.

        Returns:
        - h_tre_rnkscore: torch.Tensor, ranker scores for treatment group.
        - h_unt_rnkscore: torch.Tensor, ranker scores for control group.
        """

        # compute average (# of treatment / total # of samples) as e_hat
        # e_hat = D_tre.shape[0] / (D_tre.shape[0] + D_unt.shape[0])


        if self.num_hidden > 0:
            h_tre = torch.tanh(self.hidden_layer(D_tre))
            h_unt = torch.tanh(self.hidden_layer(D_unt))
            h_tre_rnkscore = torch.tanh(self.ranker(h_tre))
            h_unt_rnkscore = torch.tanh(self.ranker(h_unt))
        else:
            h_tre_rnkscore = torch.tanh(self.ranker(D_tre))
            h_unt_rnkscore = torch.tanh(self.ranker(D_unt))
        
        return h_tre_rnkscore, h_unt_rnkscore
    
    def soft_abs(self, x, beta=40.0):
        return torch.log(1 + torch.exp(beta * x)) + torch.log(1 + torch.exp(-beta * x)) / beta

    def compute_objective(self, h_tre_rnkscore, h_unt_rnkscore, c_tre, c_unt, o_tre, o_unt, e_hat, e_x_tre, e_x_unt):
        """
        Computes the cost-gain effectiveness objective.

        Args:
        - h_tre_rnkscore: torch.Tensor, ranker scores for treatment group.
        - h_unt_rnkscore: torch.Tensor, ranker scores for control group.
        - c_tre: torch.Tensor, cost labels for treatment group.
        - c_unt: torch.Tensor, cost labels for control group.
        - o_tre: torch.Tensor, order labels for treatment group.
        - o_unt: torch.Tensor, order labels for control group.

        Returns:
        - obj: torch.Tensor, the computed objective.
        """

        # e_x_tre = torch.clamp(e_x[treat_index], min=1e-6, max=1.0)
        # e_x_unt = torch.clamp(e_x[untreat_index], min=1e-6, max=1.0)

        s_tre = torch.exp(F.log_softmax(h_tre_rnkscore.squeeze(), dim=0))
        s_unt = torch.exp(F.log_softmax(h_unt_rnkscore.squeeze(), dim=0))


        # Compute IPS-weighted expected costs and outcomes
        dc_tre = torch.sum((s_tre * c_tre) / e_x_tre)
        dc_unt = torch.sum((s_unt * c_unt) / (1-e_x_unt))

        do_tre = torch.sum((s_tre * o_tre) / e_x_tre)
        do_unt = torch.sum((s_unt * o_unt) / (1-e_x_unt))

        # obj = (dc_tre - dc_unt) / (do_tre - do_unt)

        # Optional differentiable version:
        obj = (self.soft_abs(e_hat*do_tre - (1-e_hat)*do_unt))/(self.soft_abs(e_hat*dc_tre - (1-e_hat)*dc_unt)+1e-10)

        # obj = (F.relu(e_hat*do_tre - (1-e_hat)*do_unt))/(F.relu(e_hat*dc_tre - (1-e_hat)*dc_unt)+1e-10)
        return obj, e_hat*dc_tre - (1-e_hat)*dc_unt, e_hat*do_tre - (1-e_hat)*do_unt
    

    def optimize_model_pro(self, model, X, w, D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, lr=0.0001, epochs = 10):
        """
        Optimizes the model using the Adam optimizer.

        Args:
        - model: nn.Module, the model to optimize.
        - D_tre: torch.Tensor, features for treatment group.
        - D_unt: torch.Tensor, features for control group.
        - c_tre: torch.Tensor, cost labels for treatment group.
        - c_unt: torch.Tensor, cost labels for control group.
        - o_tre: torch.Tensor, order labels for treatment group.
        - o_unt: torch.Tensor, order labels for control group.
        - lr: float, learning rate.

        Returns:
        - obj: torch.Tensor, the computed objective.
        """
        optimizer = Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()

        e_x = torch.tensor(self.fit_predict_p_hat(X, w), dtype=torch.float32)
        # print(e_x)

        treat_index = np.where(w==1)[0]
        untreat_index = np.where(w==0)[0]

        e_hat = len(treat_index) / (len(treat_index) + len(untreat_index))
        # print(e_hat)

        e_x_tre = e_x[treat_index]
        e_x_unt = e_x[untreat_index]
        
        for epoch in range(epochs):
            h_tre_rnkscore, h_unt_rnkscore = model(D_tre, D_unt)
            obj, a, b = model.compute_objective(h_tre_rnkscore, h_unt_rnkscore, c_tre, c_unt, o_tre, o_unt, e_hat, e_x_tre, e_x_unt)

            (-obj).backward()  # Negative objective for maximization
            optimizer.step()
            
            # print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}")
            print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}, tau_C: {a.item()}, tau_O: {b.item()}")

        return obj