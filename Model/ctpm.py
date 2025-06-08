# Continuous Treatment Propensity Model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset

def soft_abs(x, beta=40.0):
    return torch.log(1 + torch.exp(beta * x)) + torch.log(1 + torch.exp(-beta * x)) / beta

class CTPM(nn.Module):
    def __init__(self, Da_dim, Db_dim, num_hidden, temp, p_quantile, dropout_rate):
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

        # Prior network
        if num_hidden > 0:
            self.hidden_prior = nn.Sequential(
                nn.Linear(Da_dim, num_hidden),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            )
            self.prior_score = nn.Linear(num_hidden, 1)
        else:
            self.prior_score = nn.Linear(Da_dim, 1)

        # Match network
        if num_hidden > 0:
            self.hidden_match = nn.Sequential(
                nn.Linear(Db_dim, num_hidden),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            )
            self.match_score = nn.Linear(num_hidden, 1)
        else:
            self.match_score = nn.Linear(Db_dim, 1)

        # Policy network on Da+Db
        if num_hidden > 0:
            self.hidden_policy = nn.Sequential(
                nn.Linear(Da_dim, num_hidden),
                nn.Tanh(),
                nn.Dropout(dropout_rate)
            )
            self.policy_score = nn.Linear(num_hidden, 1)
        else:
            self.policy_score = nn.Linear(Da_dim, 1)

        # Cosine similarity embeddings
        """
        self.match_embed_a = nn.Sequential(
            nn.Linear(Da_dim, num_hidden),
            nn.Tanh()
        )
        self.match_embed_b = nn.Sequential(
            nn.Linear(Db_dim, num_hidden),
            nn.Tanh()
        )
        """

    def forward(self, Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt):
                # d2d_tre, d2d_unt, d2dlamb):

        """
        Da_tre: treatment data, user features
        Da_unt: control data
        Db_tre: treatment data, coupon features
        Db_unt: control data
        o_tre: treatment reward
        o_unt: control reward
        c_tre: treatment cost
        c_unt: control cost
        i_tre: treatment intensity (ex. number of working hours)
        i_unt: control intensity (ex. number of working hours)
        """

        # Prior scores
        if self.num_hidden > 0:
            tre_prior_hidden = self.hidden_prior(Da_tre)
            unt_prior_hidden = self.hidden_prior(Da_unt)
            p_tre = torch.sigmoid(self.prior_score(tre_prior_hidden))
            p_unt = torch.sigmoid(self.prior_score(unt_prior_hidden))
        else:
            p_tre = torch.sigmoid(self.prior_score(tre_prior_hidden))
            p_unt = torch.sigmoid(self.prior_score(unt_prior_hidden))

        # Match scores
        #if self.num_hidden > 0:
        tre_match_hidden = self.hidden_match(Db_tre)
        unt_match_hidden = self.hidden_match(Db_unt)

        # Policy scores: prediction of intensity
        if self.num_hidden > 0:
            tre_policy_hidden = self.hidden_policy(Da_tre)
            unt_policy_hidden = self.hidden_policy(Da_unt)
            tre_policy_score = torch.sigmoid(self.policy_score(tre_policy_hidden))
            unt_policy_score = torch.sigmoid(self.policy_score(unt_policy_hidden))
        else:
            tre_policy_score = torch.sigmoid(self.policy_score(Da_tre))
            unt_policy_score = torch.sigmoid(self.policy_score(Da_unt))

        # difference between actually treatment intensity and predicted optimal intensity
        diff_tre = i_tre.unsqueeze(1) - tre_policy_score
        diff_unt = i_unt.unsqueeze(1) - unt_policy_score

        if self.eval():
            diff_tre = torch.zeros_like(diff_tre)
            diff_unt = torch.zeros_like(diff_unt)
        
        # derivative of sigmoid has a bell-shape curve
        # it peaks at diff = 0
        # decreases symmetrically as diff moves away from 0
        # it acts as a treatment intensity alignment penalty/reward
        # low score if uses are over-treated or under-treated (prediction not align with actual intensity)

        lh_tre = torch.sigmoid(diff_tre) * (1 - torch.sigmoid(diff_tre))
        lh_unt = torch.sigmoid(diff_unt) * (1 - torch.sigmoid(diff_unt))
        
        s1_tre = p_tre * lh_tre # elementwise product 
        s1_unt = p_unt * lh_unt # elementwise product 
        
        # normalize scores
        s1_tre = s1_tre / (torch.sum(s1_tre, dim=0) + 1e-17)
        s1_unt = s1_unt / (torch.sum(s1_unt, dim=0) + 1e-17)

        # Compute cosine similarity embeddings
        tre_embed_a = tre_prior_hidden
        tre_embed_b = tre_match_hidden
        unt_embed_a = unt_prior_hidden
        unt_embed_b = unt_match_hidden

        # Cosine similarity + 1
        cos_plus1_sim_tre = F.cosine_similarity(tre_embed_a, tre_embed_b, dim=1) + 1
        cos_plus1_sim_tre = cos_plus1_sim_tre.unsqueeze(1)
        cos_plus1_sim_unt = F.cosine_similarity(unt_embed_a, unt_embed_b, dim=1) + 1
        cos_plus1_sim_unt = cos_plus1_sim_unt.unsqueeze(1)

        if self.eval():
            cos_plus1_sim_tre = torch.ones_like(cos_plus1_sim_tre) + 1 
            cos_plus1_sim_unt = torch.ones_like(cos_plus1_sim_unt) + 1 

        s2_tre = s1_tre * cos_plus1_sim_tre 
        s2_unt = s1_unt * cos_plus1_sim_unt 

        s2_tre = s2_tre / (torch.sum(s2_tre, dim=0)) 
        s2_unt = s2_unt / (torch.sum(s2_unt, dim=0))
        
        # Objective

        #dc_tre = torch.sum(s_tre.float() * c_tre.float())
        #dc_unt = torch.sum(s_unt.float() * c_unt.float())
        do_tre = torch.sum(s2_tre.float() * o_tre.float())
        do_unt = torch.sum(s2_unt.float() * o_unt.float())

        # dd_tre = torch.sum(s_tre * d2d_tre)
        # dd_unt = torch.sum(s_unt * d2d_unt)

        # Cost-gain effectiveness
        #cost_diff = dc_tre - dc_unt
        order_diff = do_tre - do_unt
        # dist_diff = F.leaky_relu(dd_tre - dd_unt)

        # Objective
        # obj = order_diff / (cost_diff + 1e-9) + d2dlamb * dist_diff
        # obj = order_diff / (cost_diff + 1e-9)
        # obj = soft_abs(order_diff) / (soft_abs(cost_diff) + 1e-10)
        obj = order_diff

        return obj, s2_tre.detach(), s2_unt.detach() #, dc_tre - dc_unt, do_tre - do_unt, #h_tre_rnkscore, h_unt_rnkscore
    
def optimize_ctpm_model(model, Da_tre, Da_unt, Db_tre, Db_unt, c_tre, c_unt, o_tre, o_unt, i_tre, i_unt, batch_size = 8000, lr=0.001, epochs = 10):
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

    # Dataset
    tre_dataset = TensorDataset(Da_tre, Db_tre, o_tre, c_tre, i_tre)
    unt_dataset = TensorDataset(Da_unt, Db_unt, o_unt, c_unt, i_unt)

    tre_loader = DataLoader(tre_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    unt_loader = DataLoader(unt_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        for tre_batch, unt_batch in zip(tre_loader, unt_loader):
            optimizer.zero_grad()
            Da_tre, Db_tre, o_tre, c_tre, i_tre = tre_batch
            Da_unt, Db_unt, o_unt, c_unt, i_unt = unt_batch
            obj, _, _ = model(Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt)
            (-obj).backward()  # Negative objective for maximization
            optimizer.step()
            print(f"Epoch {epoch}/{epochs}, Objective: {obj}")
    return obj
# we use policy network to compute bell shape which later is used to compute p on top of the match network that produces p