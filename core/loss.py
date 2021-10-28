import numpy as np
import torch

device='cuda'
def np2tc(x_np): return torch.from_numpy(x_np).float().to(device)
def tc2np(x_tc): return x_tc.detach().cpu().numpy()

def mln_gather(pi,mu,sigma):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    max_idx = torch.argmax(pi,dim=1) # [N]
    mu      = torch.softmax(mu,dim=2) #[N x K x D]
    idx_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu,dim=1,index=idx_gather).squeeze(dim=1) # [N x D]
    sigma_sel = torch.gather(sigma,dim=1,index=idx_gather).squeeze(dim=1) # [N x D]
    out = {'max_idx':max_idx, # [N]
           'idx_gather':idx_gather, # [N x 1 x D]
           'mu_sel':mu_sel, # [N x D]
           'sigma_sel':sigma_sel # [N x D]
           }
    return out

def mln_eval(pi,mu,sigma,num,N=10):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    top_pi,top_idx = torch.topk(pi,num,dim=1) # [N X n]
    top_pi=torch.softmax(top_pi,dim=-1)
    max_idx = torch.argmax(pi,dim=1) # [N]
    max2_idx= top_idx[:,1] # [N]

    mu      = torch.softmax(mu,dim=2) # [N x K x D]
    mu_max = torch.argmax(mu,dim=2) # [N x K]
    mu_onehot=_to_one_hot(mu_max,N)

    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]

    mu_exp = torch.mul(pi_exp,mu_onehot) # mixtured mu [N x K x D]
    mu_prime = torch.sum(mu_exp,dim=1) # [N x D]

    sig_exp = torch.mul(pi_exp,sigma) # mixtured mu [N x K x D]
    sig_prime = torch.sum(sig_exp,dim=1) # [N x D]

    idx1_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu,dim=1,index=idx1_gather).squeeze(dim=1) # [N x D]
    sigma_sel = torch.gather(sigma,dim=1,index=idx1_gather).squeeze(dim=1) # [N x D]

    idx2_gather = max2_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel2 = torch.gather(mu,dim=1,index=idx2_gather).squeeze(dim=1) # [N x D]
    sigma_sel2 = torch.gather(sigma,dim=1,index=idx2_gather).squeeze(dim=1) # [N x D]
   
    unct_out = mln_uncertainties(pi,mu,sigma)
    pi_entropy = unct_out['pi_entropy'] # [N]

    out = {'max_idx':max_idx, # [N]
           'mu_sel':mu_sel, # [N x D]
           'sigma_sel':sigma_sel, # [N x D]
           'mu_sel2':mu_sel2, # [N x D]
           'sigma_sel2':sigma_sel2, # [N x D]
           'mu_prime': mu_prime, # [N x D]
           'sigma_prime': sig_prime, # [N x D]
           'pi_entropy': pi_entropy, # [N]
           'top_pi':top_pi
           }
    return out

def mace_loss(pi,mu,sigma,target):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
        :param target:  [N x D]
    """
    # $\mu$
    mu_hat = torch.softmax(mu,dim=2) # logit to prob [N x K x D]
    log_mu_hat = torch.log(mu_hat+1e-6) # [N x K x D]
    # $\pi$
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(mu) # [N x K x D]
    # target
    target_usq =  torch.unsqueeze(target,1) # [N x 1 x D]
    target_exp =  target_usq.expand_as(mu) # [N x K x D]
    # CE loss
    ce_exp = -target_exp*log_mu_hat # CE [N x K x D]
    ace_exp = ce_exp / sigma # attenuated CE [N x K x D]
    mace_exp = torch.mul(pi_exp,ace_exp) # mixtured attenuated CE [N x K x D]
    ce_exp = torch.mul(pi_exp,ce_exp)
    ce=torch.sum(ce_exp,dim=1) # [N x D]
    ce=torch.sum(ce,dim=1) # [N]
    mace = torch.sum(mace_exp,dim=1) # [N x D]
    mace = torch.sum(mace,dim=1) # [N]
    mace_avg = torch.mean(mace) # [1]
    ce_avg=torch.mean(ce) # [1]
    # Compute uncertainties (epis and alea)
    unct_out = mln_uncertainties(pi,mu,sigma)
    epis = unct_out['epis'] # [N]
    alea = unct_out['alea'] # [N]
    pi_entropy = unct_out['pi_entropy'] # [N]
    epis_avg = torch.mean(epis) # [1]
    alea_avg = torch.mean(alea) # [1]
    pi_entropy_avg=torch.mean(pi_entropy) # [1]
    # Return
    loss_out = {'mace':mace, # [N]
                'ce_avg':ce_avg,
                'mace_avg':mace_avg, # [1]
                'epis':epis, # [N]
                'alea':alea, # [N]
                'epis_avg':epis_avg, # [1]
                'alea_avg':alea_avg, # [1],
                'pi_entropy':pi_entropy, # [N]
                'pi_entropy_avg':pi_entropy_avg # [1]
                }
    return loss_out

def kendal_loss(pi,mu,sigma,target):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
        :param target:  [N x D]
    """
    # $\mu$
    #N = TD.Normal(mu, sigma)
    mu_hat = torch.softmax(mu + sigma*torch.randn_like(sigma),dim=2)
    #mu_hat = torch.softmax(mu,dim=2) # logit to prob [N x K x D]
    log_mu_hat = torch.log(mu_hat+1e-6) # [N x K x D]
    # $\pi$
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(mu) # [N x K x D]
    # target
    target_usq =  torch.unsqueeze(target,1) # [N x 1 x D]
    target_exp =  target_usq.expand_as(mu) # [N x K x D]
    # CE loss
    ce_exp = -target_exp*log_mu_hat # CE [N x K x D]
    ace_exp = ce_exp #/ sigma # attenuated CE [N x K x D]
    mace_exp = torch.mul(pi_exp,ace_exp) # mixtured attenuated CE [N x K x D]
    mace = torch.sum(mace_exp,dim=1) # [N x D]
    mace = torch.sum(mace,dim=1) # [N]
    mace_avg = torch.mean(mace) # [1]
    # Compute uncertainties (epis and alea)
    unct_out = mln_uncertainties(pi,mu,sigma)
    epis = unct_out['epis'] # [N]
    alea = unct_out['alea'] # [N]
    pi_entropy = unct_out['pi_entropy']
    epis_avg = torch.mean(epis) # [1]
    alea_avg = torch.mean(alea) # [1]
    pi_entropy_avg=torch.mean(pi_entropy)
    # Return
    loss_out = {'mace':mace, # [N]
                'mace_avg':mace_avg, # [1]
                'epis':epis, # [N]
                'alea':alea, # [N]
                'epis_avg':epis_avg, # [1]
                'alea_avg':alea_avg # [1]
                }
    return loss_out

def mln_uncertainties(pi,mu,sigma):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) #[N]
    # $\pi$
    mu_hat = torch.softmax(mu,dim=2) # logit to prob [N x K x D]
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]
    # softmax($\mu$) average
    mu_hat_avg = torch.sum(torch.mul(pi_exp,mu_hat),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_hat_avg_exp = mu_hat_avg.expand_as(mu) # [N x K x D]
    mu_hat_diff_sq = torch.square(mu_hat-mu_hat_avg_exp) # [N x K x D]
    # Epistemic uncertainty
    epis = torch.sum(torch.mul(pi_exp,mu_hat_diff_sq), dim=1)  # [N x D]
    epis = torch.sqrt(torch.sum(epis,dim=1)+1e-6) # [N]
    # Aleatoric uncertainty
    alea = torch.sum(torch.mul(pi_exp,sigma), dim=1)  # [N x D]
    alea = torch.sqrt(torch.mean(alea,dim=1)+1e-6) # [N]
    top_pi,top_idx = torch.topk(pi,2,dim=1) # [N X 2]
    # Return
    unct_out = {'epis':epis, # [N]
                'alea':alea,  # [N]
                'pi_entropy':entropy_pi,'top_pi':top_pi
                }
    return unct_out

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(device)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)
