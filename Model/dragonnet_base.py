import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ziln import zero_inflated_lognormal_pred, zero_inflated_lognormal_loss


class DragonNetBase(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super(DragonNetBase, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=shared_hidden)
        self.fc2 = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)
        self.fcz = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)

        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)

        self.y0_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y0_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y0_out = nn.Linear(in_features=outcome_hidden, out_features=3)

        self.y1_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y1_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y1_out = nn.Linear(in_features=outcome_hidden, out_features=3)

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fcz(x))

        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.relu(self.y0_fc1(z))
        y0 = F.relu(self.y0_fc2(y0))
        y0 = self.y0_out(y0)

        y1 = F.relu(self.y1_fc1(z))
        y1 = F.relu(self.y1_fc2(y1))
        y1 = self.y1_out(y1)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps

def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
#    print(t_true.shape)
#    print(y0_pred.shape)
#    print(y1_pred.shape)
    # Ensure correct dimensions
    y_true = y_true.squeeze(-1)
    t_true = t_true.squeeze(-1)
    
    y0_pred = zero_inflated_lognormal_pred(y0_pred).squeeze(-1)
    y1_pred = zero_inflated_lognormal_pred(y1_pred).squeeze(-1)

    tau_pred = y1_pred - y0_pred

    # Index using 1D mask
    treated_mask = t_true == 1.0
    control_mask = t_true == 0.0

    tau_pred_t = tau_pred[treated_mask].unsqueeze(1)  # shape: [N1, 1]
    tau_pred_c = tau_pred[control_mask].unsqueeze(1)  # shape: [N0, 1]

    softmax_tau_t = F.softmax(tau_pred_t, dim=0)
    softmax_tau_c = F.softmax(tau_pred_c, dim=0)

    treated_y = y_true[treated_mask].unsqueeze(1)
    control_y = y_true[control_mask].unsqueeze(1)

    N1 = treated_y.shape[0]
    N0 = control_y.shape[0]

    if N1 == 0 or N0 == 0:
        return torch.tensor(0.0, requires_grad=True)  # Avoid crash if a minibatch is all treated or all control

    loss = - (N1 + N0) * (
        (1 / N1) * torch.sum(treated_y * torch.log(softmax_tau_t + 1e-8)) -
        (1 / N0) * torch.sum(control_y * torch.log(softmax_tau_c + 1e-8))
    )

    return loss


def outcome_ranking_loss(y_true, t_true, y0_pred, y1_pred):
    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
    # enchance the rankability of outcome regression inner the treatment/control group 
    treated_y = y_true[t_true==1.0].unsqueeze(1)
    control_y = y_true[t_true==0.0].unsqueeze(1)  
      
    outputs_h1 = y1_pred[t_true==1.0].unsqueeze(1)
    outputs_h0 = y0_pred[t_true==0.0].unsqueeze(1) 

    # pairwise outcome ranking loss in treatment group
    """
    treat_loss = torch.tensor([0.0])
    for i in torch.randint(low=0,high=outputs_h1.shape[0],size=(100,)):
        for j in torch.randint(low=0,high=outputs_h1.shape[0],size=(100,)):
            pair_loss = ((outputs_h1[i] - outputs_h1[j]) - (treated_y[i] - treated_y[j])) ** 2 if (outputs_h1[i] - outputs_h1[j]) * (treated_y[i] - treated_y[j]) < 0 else torch.tensor([0.0]).to(config.device)
            treat_loss += pair_loss
    """    

    outputs_h1_matrix = outputs_h1 - outputs_h1.T
    treated_y_matrix = treated_y - treated_y.T
    product = outputs_h1_matrix * treated_y_matrix
    new_tensor = torch.zeros_like(outputs_h1_matrix)
    mask = product >= 0
    new_tensor = (outputs_h1_matrix - treated_y_matrix) ** 2
    new_tensor[mask] = 0.0
    treat_loss = torch.sum(new_tensor)
    


    # pairwise outcome ranking loss in control group
    """
    control_loss = torch.tensor([0.0])
    for i in torch.randint(low=0,high=outputs_h0.shape[0],size=(100,)):
        for j in torch.randint(low=0,high=outputs_h0.shape[0],size=(100,)):
            pair_loss = ((outputs_h0[i] - outputs_h0[j]) - (control_y[i] - control_y[j])) ** 2 if (outputs_h0[i] - outputs_h0[j]) * (control_y[i] - control_y[j]) <     0 else torch.tensor([0.0]).to(config.device)
            control_loss += pair_loss
    """    

    outputs_h0_matrix = outputs_h0 - outputs_h0.T
    control_y_matrix = control_y - control_y.T
    product = outputs_h0_matrix * control_y_matrix
    new_tensor = torch.zeros_like(outputs_h0_matrix)
    mask = product >= 0
    new_tensor = (outputs_h0_matrix - control_y_matrix) ** 2
    new_tensor[mask] = 0.0
    control_loss = torch.sum(new_tensor)
    

    return treat_loss + control_loss

 
def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, ranking_lambda=1.0):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """

    t_pred = (t_pred + 0.01) / 1.02
    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    #loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    #loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss0 = torch.sum((1. - t_true) * zero_inflated_lognormal_loss(y_true, y0_pred))
    loss1 = torch.sum(t_true * zero_inflated_lognormal_loss(y_true, y1_pred))
    
    loss_uplift_ranking = uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred)
    loss_outcome_ranking = outcome_ranking_loss(y_true, t_true, y0_pred, y1_pred)
    print('loss_uplift_ranking', loss_uplift_ranking)
    print('loss_outcome_ranking', loss_outcome_ranking)
    loss_y = loss0 + loss1 + 10 * loss_uplift_ranking  + 1e-4 * loss_outcome_ranking #loss_uplift_ranking: 10, loss_outcome_ranking: 1e-4

    loss = loss_y + alpha * loss_t

    return loss


def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, ranking_lambda, alpha=1.0, beta=1.0):
    """
    Targeted regularisation loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        targeted regularization hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha, ranking_lambda)
    t_pred = (t_pred + 0.01) / 1.02

    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = torch.sum((y_true - y_pert)**2)

    # final
    loss = vanilla_loss + beta * targeted_regularization
    return loss


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

