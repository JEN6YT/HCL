import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class percentile_barrier_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, initial_temp, p_quantile):
        """
        Initializes the Direct Ranking Model.

        Args:
        - input_dim: int, dimensionality of user context features (both treatment and control).
        - hidden_dim: int, number of hidden units. If 0, skips the hidden layer.
        - initial_temp: float, initial temperature (this is tunable) of the sigmoid governing p_quantile cut-off
        - p_quantile: tensor, the top-p-quantile number between (0, 1) 
        """
        super(percentile_barrier_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.p_quantile = p_quantile
        self.temperature = initial_temp
        
        if hidden_dim > 0:
            self.hidden_layer = nn.Linear(input_dim, hidden_dim)
            self.ranker = nn.Linear(hidden_dim, 1)
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
        if self.hidden_dim > 0:
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

def compute_objective_pb(D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, model):
    
    # Forward pass: compute rank scores
    h_tre_rnkscore, h_unt_rnkscore= model(D_tre, D_unt)
    
    s_tre = F.softmax(h_tre_rnkscore.squeeze(), dim=0)
    s_unt = F.softmax(h_unt_rnkscore.squeeze(), dim=0)
    
    # Sort scores
    h_tre_sorted, _ = torch.sort(s_tre, descending=True, dim=0)
    h_unt_sorted, _ = torch.sort(s_unt, descending=True, dim=0)
    
    # Top-k selection
    top_k_tre = int(torch.ceil(h_tre_sorted.size(0) * model.p_quantile).item())
    top_k_unt = int(torch.ceil(h_unt_sorted.size(0) * model.p_quantile).item())
    
    intercept_tre = h_tre_sorted[top_k_tre - 1]
    intercept_unt = h_unt_sorted[top_k_unt - 1]
    
    # Apply sigmoid with temperature
    h_tre = torch.sigmoid(model.temperature * (s_tre - intercept_tre))
    h_unt = torch.sigmoid(model.temperature * (s_unt - intercept_unt))
    
    # Compute weighted sums for treatment effects
    dc_tre = torch.sum(s_tre * h_tre * c_tre)
    dc_unt = torch.sum(s_unt * h_unt * c_unt)
    do_tre = torch.sum(s_tre * h_tre * o_tre)
    do_unt = torch.sum(s_unt * h_unt * o_unt)
    
    # Compute the objective
    #obj = F.relu(dc_tre - dc_unt) / (F.relu(do_tre - do_unt) + 1e-7)  # Adding a small epsilon to avoid division by zero
    obj = soft_abs(do_tre - do_unt) / (soft_abs(dc_tre - dc_unt) + 1e-10)
    # obj = (do_tre - do_unt) / (dc_tre - dc_unt + 1e-9) 
    return obj, dc_tre - dc_unt, do_tre - do_unt


def optimize_model_pb(model, D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, lr=0.001, epochs = 10):
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
        if epoch % 100 == 0: 
            model.temperature += 0.1 
        obj, a, b = compute_objective_pb(D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, model)
        (-obj).backward()  # Negative objective for maximization
        optimizer.step()
        print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}, tau_C: {a.item()}, tau_O: {b.item()}")
        #print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}")
    return obj

"""
import numpy as np
import tensorflow as tf

def TunableTQRankingModelDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, initial_temp, p_quantile, num_hidden, use_schedule=False): 
    ## implements the top-p-quantile operator for Constrained Ranking Model 
    ## with tunable temperature through gradient descent and temperature schedule 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## p_quantile: the top-p-quantile number between (0, 1) 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    ## initial_temp: initial temperature (this is tunable) of the sigmoid governing p_quantile cut-off 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## Consider the contrast TQRanking Model 
    ## as opposed to improving cpit upon control cohort, 
    ## [E(Ctqr) - E(Cctrl)] / [E(Ttqr) - E(Tctrl)] 
    ## let's think about improving cpit upon treatment cohort 
    ## [E(Ctqr) - E(Ctre)] / [E(Ttqr) - E(Ttre)] 
    ## or, let's think about improving upon DRM 
    ## 
    
    ## temperature of the sigmoid governing p_quantile cut-off 
    with graph.as_default() as g: 
        #if use_schedule == False: 
        init = tf.constant(initial_temp, dtype=tf.float64) 
        with tf.variable_scope("temp", reuse=tf.AUTO_REUSE) as scope: 
            temp = tf.get_variable('temp', initializer=init, dtype=tf.float64) 
        ### ---- the following code makes the temperature tunable ---- 
        ### deleted for use of temperature schedule, but keep for future applications 
        #else: 
        #    temp = tf.constant(initial_temp, dtype=tf.float64) 
            #tf.Variable(2.5, dtype=tf.float64, trainable=True) 
        #init2 = tf.constant(p_quantile, dtype=tf.float64) 
        #with tf.variable_scope("p_quantile", reuse=tf.AUTO_REUSE) as scope: 
        #    p_quantile = tf.get_variable('p_quantile', initializer=init2, dtype=tf.float64)
            #tf.Variable(0.3, dtype=tf.float32, trainable=True, reuse=tf.AUTO_REUSE)
        
        ## define size of cohort datasets 
        size_tre = D_tre.shape[0] 
        size_unt = D_unt.shape[0] 
        
        ### ----- define model graph of Top Quantile Constrained ranking ----- 

        ### we can define either a linear or a multi-layer neural network 
        ### for the ranker or scorer 
        if num_hidden > 0: 
            with tf.variable_scope("tqrhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h1_unt = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("tqranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(h1_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        else: 
            with tf.variable_scope("tqranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        
        ### adopt a sorting operator that's also differentiable 
        ### for application of back-propagation and gradient optimization 
        h_tre_sorted = tf.contrib.framework.sort(h_tre_rnkscore, axis=0, direction='DESCENDING') 
        h_unt_sorted = tf.contrib.framework.sort(h_unt_rnkscore, axis=0, direction='DESCENDING') 
        
        top_k_tre = tf.cast(tf.ceil(size_tre * p_quantile), tf.int32) 
        top_k_unt = tf.cast(tf.ceil(size_unt * p_quantile), tf.int32) 
        
        intercept_tre = tf.slice(h_tre_sorted, [top_k_tre - 1, 0], [1, 1]) 
        intercept_unt = tf.slice(h_unt_sorted, [top_k_unt - 1, 0], [1, 1]) 
        
        ### stop gradients at the tunable intercept for sigmoid 
        ### to stabilize gradient-based optimization 
        intercept_tre = tf.stop_gradient(intercept_tre) 
        intercept_unt = tf.stop_gradient(intercept_unt) 
        
        ### use sigmoid to threshold top-k candidates, or use more sophisticated hinge loss 
        h_tre = tf.sigmoid(temp * (h_tre_rnkscore - intercept_tre)) 
        h_unt = tf.sigmoid(temp * (h_unt_rnkscore - intercept_unt)) 
        
        ### using softmax and weighted reduce-sum to compute the expected value 
        ### of treatment effect functions 
        s_tre = tf.nn.softmax(h_tre, axis=0) 
        s_unt = tf.nn.softmax(h_unt, axis=0) 
        
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
    
    return obj, opt, h_tre_rnkscore, h_unt_rnkscore, temp, p_quantile 
"""