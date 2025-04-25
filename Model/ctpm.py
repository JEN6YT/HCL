# Continuous Treatment Propensity Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam 

class CTPM(nn.Module):
    def __init__(self, D_dim, num_hidden, temp, p_quantile, dropout_rate):
        """
        D_dim: feature dimension
        num_hidden: number of hidden units in the network
        temp: temperature used to sharpen or smooth the sigmoid function for ranking
        """
        super(CTPM, self).__init__()
        self.num_hidden = num_hidden
        self.temp = temp
        self.p_quantile = p_quantile
        self.dropout_rate = dropout_rate

        # Match network
        if num_hidden > 0:
            self.hidden_match = nn.Sequential(
                nn.Linear(D_dim, num_hidden),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            )
            self.match_score = nn.Linear(num_hidden, 1)
        else:
            self.match_score = nn.Linear(D_dim, 1)

        # Policy network on Da+Db
        if num_hidden > 0:
            self.hidden_policy = nn.Sequential(
                nn.Linear(D_dim, num_hidden),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            )
            self.policy_score = nn.Linear(num_hidden, 1)
        else:
            self.policy_score = nn.Linear(D_dim, 1)

    def forward(self, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt):
                # d2d_tre, d2d_unt, d2dlamb):

        """
        D_tre: treatment data
        D_unt: control data
        o_tre: treatment reward
        o_unt: control reward
        c_tre: treatment cost
        c_unt: control cost
        i_tre: treatment intensity (ex. number of working hours)
        i_unt: control intensity (ex. number of working hours)
        """

        size_tre = D_tre.shape[0]
        size_unt = D_unt.shape[0]

        # Match scores
        if self.num_hidden > 0:
            tre_match_hidden = self.hidden_match(D_tre)
            unt_match_hidden = self.hidden_match(D_unt)
            s_tre = torch.sigmoid(self.match_score(tre_match_hidden))
            s_unt = torch.sigmoid(self.match_score(unt_match_hidden))
        else:
            s_tre = torch.sigmoid(self.match_score(D_tre))
            s_unt = torch.sigmoid(self.match_score(D_unt))

        # Policy scores: prediction of intensity
        if self.num_hidden > 0:
            tre_policy_hidden = self.hidden_policy(D_tre)
            unt_policy_hidden = self.hidden_policy(D_unt)
            tre_policy_score = torch.sigmoid(self.policy_score(tre_policy_hidden))
            unt_policy_score = torch.sigmoid(self.policy_score(unt_policy_hidden))
        else:
            tre_policy_score = torch.sigmoid(self.policy_score(D_tre))
            unt_policy_score = torch.sigmoid(self.policy_score(D_unt))


        # Bell-shaped intensity score

        # difference between actually treatment intensity and predicted optimal intensity
        diff_tre = i_tre.unsqueeze(1) - tre_policy_score
        diff_unt = i_unt.unsqueeze(1) - unt_policy_score

        # derivative of sigmoid has a bell-shape curve
        # it peaks at diff = 0
        # decreases symmetrically as diff moves away from 0
        # it acts as a treatment intensity alignment penalty/reward
        # low score if uses are over-treated or under-treated (prediction not align with actual intensity)
        lh_tre = torch.sigmoid(diff_tre) * (1 - torch.sigmoid(diff_tre))
        lh_unt = torch.sigmoid(diff_unt) * (1 - torch.sigmoid(diff_unt))

        s_tre = s_tre * lh_tre
        s_unt = s_unt * lh_unt
        s_tre = s_tre /(s_tre.sum() + 1e-17)
        s_unt = s_unt /(s_unt.sum() + 1e-17)

        h_tre_rnkscore, h_unt_rnkscore = s_tre.clone(), s_unt.clone()

        # Top-k sorting with differentiable approximation
        top_k_tre = int(torch.ceil(torch.tensor(size_tre * self.p_quantile)).item())
        top_k_unt = int(torch.ceil(torch.tensor(size_unt * self.p_quantile)).item())

        h_tre_sorted, _ = torch.sort(s_tre, dim=0, descending=True)
        h_unt_sorted, _ = torch.sort(s_unt, dim=0, descending=True)

        intercept_tre = h_tre_sorted[top_k_tre - 1].detach()
        intercept_unt = h_unt_sorted[top_k_unt - 1].detach()

        h_tre = torch.sigmoid(self.temp * (s_tre - intercept_tre))
        h_unt = torch.sigmoid(self.temp * (s_unt - intercept_unt))

        h_tre = F.dropout(h_tre, self.dropout_rate)
        h_unt = F.dropout(h_unt, self.dropout_rate)

        # Softmax weights
        s_tre = F.softmax(h_tre, dim=0)
        s_unt = F.softmax(h_unt, dim=0)

        # Objective
        dc_tre = torch.sum(s_tre.float() * c_tre.float())
        dc_unt = torch.sum(s_unt.float() * c_unt.float())
        do_tre = torch.sum(s_tre.float() * o_tre.float())
        do_unt = torch.sum(s_unt.float() * o_unt.float())
        # dd_tre = torch.sum(s_tre * d2d_tre)
        # dd_unt = torch.sum(s_unt * d2d_unt)

        # Cost-gain effectiveness
        cost_diff = F.leaky_relu(dc_tre - dc_unt)
        order_diff = F.leaky_relu(do_tre - do_unt)
        # dist_diff = F.leaky_relu(dd_tre - dd_unt)

        # Objective
        # obj = cost_diff / (order_diff + 1e-9) + d2dlamb * dist_diff
        obj = cost_diff / (order_diff + 1e-9)

        return obj, dc_tre - dc_unt, do_tre - do_unt, h_tre_rnkscore, h_unt_rnkscore
    
def optimize_ctpm_model(model, D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, i_tre, i_unt, lr=0.001, epochs = 10):
    """
    Optimizes the model using the Adam optimizer.

    Args:
    - model: nn.Module, the model to optimize.
    - D_tre: features for treatment group.
    - D_unt: features for control group.
    - c_tre: cost labels for treatment group.
    - c_unt: cost labels for control group.
    - o_tre: order labels for treatment group.
    - o_unt: order labels for control group.
    - i_tre: treatment intensity
    - i_unt: control intensity
    - lr: float, learning rate.

    Returns:
    - obj: the computed objective.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    for epoch in range(epochs):
        obj, a, b, _, _ = model(D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt)
        (-obj).backward()  # Negative objective for maximization
        optimizer.step()

        print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}, tau_C: {a.item()}, tau_O: {b.item()}")
    return obj


# we use policy network to compute bell shape which later is used to compute p on top of the match network that produces p