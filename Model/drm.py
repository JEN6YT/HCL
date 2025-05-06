import torch, pdb
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

    # obj = (do_tre - do_unt) / (dc_tre - dc_unt + 1e-9) 

    # Optional differentiable version:
    
    obj = (do_tre - do_unt) / ((dc_tre - dc_unt) + 1e-10)
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
        model.train()
        h_tre_rnkscore, h_unt_rnkscore = model(D_tre, D_unt)
        obj, a, b = compute_objective(h_tre_rnkscore, h_unt_rnkscore, c_tre, c_unt, o_tre, o_unt)
        
        (-obj).backward()  # Negative objective for maximization
        optimizer.step()
        
        print(f"Epoch {epoch}/{epoch}, Objective: {obj.item()}, tau_C: {a.item()}, tau_O: {b.item()}")
    return obj
