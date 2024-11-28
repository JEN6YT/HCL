import numpy as np, tensorflow as tf

def SimpleTCModelDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
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