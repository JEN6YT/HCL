library(grf) 
library(dplyr) 
library(data.table) 

num_trees = 100
p_alpha = 0.2 #0.2 
p_min_node_size = 5 #3
p_sample_fraction = 0.5 #0.5
num_features = 4 
mtry = 2

print('reading data from csv') 
dread <- read.csv(file='/Users/jenniferzhang/Desktop/Research with Will/HCL project/ponpare_tr.csv')

print('performing data transformations') 
features <- select(dread, 'V1', 'V2', 'V3', 'V4')
w <- select(dread, 'w_tr') 
o <- select(dread, 'values_tr') 
c <- select(dread, 'cost_tr') 

print(features) 

X <- matrix(as.vector(t(features)), , num_features, byrow = TRUE) 
W<- matrix(as.vector(t(w)), , 1) 
Y<- matrix(as.vector(t(o)), , 1) 
C<- matrix(as.vector(t(c)), , 1) 

print('fitting causal random forest for trips') 
tauO.forest <- causal_forest(X, Y, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction, mtry = mtry, tune.parameters="all") #
#tauO.forest$tuning.output 

print('fitting causal random forest for cost') 
tauC.forest <- causal_forest(X, C, W, num.trees=num_trees, alpha=p_alpha, min.node.size=p_min_node_size, sample.fraction=p_sample_fraction, mtry = mtry, tune.parameters="all") #
#tauC.forest$tuning.output 

print('reading test data from csv, data transformations') 
dread <- read.csv(file='/Users/jenniferzhang/Desktop/Research with Will/HCL project/ponpare_va.csv')

features <- select(dread, 'V1', 'V2', 'V3', 'V4') 
w <- select(dread, 'w_va') 
o <- select(dread, 'values_va') 
c <- select(dread, 'cost_va') 

Xtest <- matrix(as.vector(t(features)), , num_features, byrow = TRUE) 

print('performing prediction') 
tauO.hat <- predict(tauO.forest, Xtest) 
tauC.hat <- predict(tauC.forest, Xtest) 

print('predicting and writing to csv, value')

tdfO <- as.data.frame(t(tauO.hat)) 
write.csv(tdfO, file = paste('/Users/jenniferzhang/Desktop/Research with Will/HCL project/HCL/results_ponpare/causal_forest_grf_test_set_results_O', '_numtrees', toString(num_trees), '_alpha', toString(p_alpha), '_min_node_size', toString(p_min_node_size), '_sample_fraction', toString(p_sample_fraction),'.csv', sep='')) 

print('predicting and writing to csv, cost') 
tdfC <- as.data.frame(t(tauC.hat)) 
write.csv(tdfC, file = paste('/Users/jenniferzhang/Desktop/Research with Will/HCL project/HCL/results_ponpare/causal_forest_grf_test_set_results_C', '_numtrees', toString(num_trees), '_alpha', toString(p_alpha), '_min_node_size', toString(p_min_node_size), '_sample_fraction', toString(p_sample_fraction),'.csv', sep='')) 


