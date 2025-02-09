import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SimpleTCModelDNN(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initializes the Direct Ranking Model.

        Args:
        - input_dim: int, dimensionality of user context features (both treatment and control).
        - num_hidden: int, number of hidden units. If 0, skips the hidden layer.
        """
        super(SimpleTCModelDNN, self).__init__()
        self.num_hidden = num_hidden

        if num_hidden > 0:
            self.hidden_layer = nn.Linear(input_dim, num_hidden)
            self.ranker = nn.Linear(num_hidden, 1)
        else:
            self.ranker = nn.Linear(input_dim, 1)

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
        if self.num_hidden > 0:
            h_tre = torch.tanh(self.hidden_layer(D_tre))
            h_unt = torch.tanh(self.hidden_layer(D_unt))
            h_tre_rnkscore = torch.tanh(self.ranker(h_tre))
            h_unt_rnkscore = torch.tanh(self.ranker(h_unt))
        else:
            h_tre_rnkscore = torch.tanh(self.ranker(D_tre))
            h_unt_rnkscore = torch.tanh(self.ranker(D_unt))
        
        return h_tre_rnkscore, h_unt_rnkscore

def soft_abs(x, beta=40.0):
    return torch.log(1 + torch.exp(beta * x)) + torch.log(1 + torch.exp(-beta * x)) / beta

def compute_objective(h_tre_rnkscore, h_unt_rnkscore, c_tre, c_unt, o_tre, o_unt):
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
    s_tre = F.softmax(h_tre_rnkscore.squeeze(), dim=0)
    s_unt = F.softmax(h_unt_rnkscore.squeeze(), dim=0)

    
    dc_tre = torch.sum(s_tre * c_tre)
    dc_unt = torch.sum(s_unt * c_unt)

    do_tre = torch.sum(s_tre * o_tre)
    do_unt = torch.sum(s_unt * o_unt)

    #obj = (do_tre - do_unt) / (dc_tre - dc_unt) 

    # Optional differentiable version:
    obj = soft_abs(do_tre - do_unt) / (soft_abs(dc_tre - dc_unt) + 1e-10)    
    #obj = F.relu(do_tre - do_unt) / (F.relu(dc_tre - dc_unt))
    #obj = - F.relu(abs(dc_tre - dc_unt)) / F.relu(abs(do_tre - do_unt)) 
    #obj = - abs(dc_tre - dc_unt) / abs(do_tre - do_unt)
    return obj, dc_tre - dc_unt, do_tre - do_unt


def optimize_model(model, D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, lr=0.001, epochs = 10):
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
    
    for epoch in range(epochs):
        h_tre_rnkscore, h_unt_rnkscore = model(D_tre, D_unt)
        obj, a, b = compute_objective(h_tre_rnkscore, h_unt_rnkscore, c_tre, c_unt, o_tre, o_unt)

        (-obj).backward()  # Negative objective for maximization
        optimizer.step()
        
        print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}, tau_C: {a.item()}, tau_O: {b.item()}")
    return obj


# Example usage:
# Assuming D_tre, D_unt, c_tre, c_unt, o_tre, o_unt are PyTorch tensors
# and input_dim is the number of features for each user

'''
import numpy as np, tensorflow as tf

def SimpleTCModelDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, num_hidden): 
    
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("drmhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h1_unt = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(h1_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        
        ### use softmax normalization and weighted reduce-sum for 
        ### compute of expected value of treatment effects 
        s_tre = tf.nn.softmax(h_tre_rnkscore, axis=0) 
        s_unt = tf.nn.softmax(h_unt_rnkscore, axis=0) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt)         
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        saver = tf.compat.v1.train.Saver()         
        return obj, opt, h_tre_rnkscore, h_unt_rnkscore, saver


'''