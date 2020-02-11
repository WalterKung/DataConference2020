import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)            
            self.rho.weight = nn.Parameter(torch.randn_like(self.rho.weight))
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
        self.alphas.weight = nn.Parameter(torch.randn_like(self.alphas.weight)*0.1)
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act
        )         
        self.q_theta[0].weight = nn.Parameter(torch.randn_like(self.q_theta[0].weight) * 0.01)  
        self.q_theta[0].bias = nn.Parameter(torch.randn_like(self.q_theta[0].bias) * 0.01)
        self.q_theta[2].weight = nn.Parameter(torch.randn_like(self.q_theta[2].weight)* 0.01)  
        self.q_theta[2].bias = nn.Parameter(torch.randn_like(self.q_theta[2].bias) * 0.01)
        
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True) 
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        
        self.mu_q_theta.weight = nn.Parameter(torch.randn_like(self.mu_q_theta.weight) * 0.01)  
        self.mu_q_theta.bias = nn.Parameter(torch.randn_like(self.mu_q_theta.bias) * 0.333)
        
        self.logsigma_q_theta.weight = nn.Parameter(torch.randn_like(self.logsigma_q_theta.weight)*0.001)  
        self.logsigma_q_theta.bias = nn.Parameter(torch.randn_like(self.logsigma_q_theta.bias)*0.1)
        

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows, enc_drop = None):                        
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        if enc_drop is None: enc_drop = self.enc_drop
        q_theta = self.q_theta(bows)
        if enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows, enc_drop = None):
        if enc_drop is None: enc_drop = self.enc_drop
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows, enc_drop)
        z = self.reparameterize(mu_theta, logsigma_theta)
#        theta = torch.sigmoid(z) 
#        print("max: " + str(theta.max() ) + " min: " + str(theta.min()))
        theta = F.softmax(z, dim=-1) 

        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

