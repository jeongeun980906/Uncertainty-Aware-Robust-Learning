import torch
from core.loss import *
import matplotlib.pyplot as plt
import numpy as np

def func_eval(model,data_iter,data_size,device):
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum,entropy_pi_sum,top2_pi_sum = 0,0,0,0,0,0
        y_probs= list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            y_trgt      = batch_out.to(device)
            if data_size is None:
                mln_out     = model.forward(batch_in.to(device))
            else:
                mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [B x N]

            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]            entropy_pi  = -pi*torch.log(pi)
            entropy_pi  = unct_out['pi_entropy']
            top_pi = unct_out['top_pi'][:,1]
            entropy_pi_sum  += torch.sum(entropy_pi)
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            top2_pi_sum+=torch.sum(top_pi)
            # Check predictions
            y_prob,y_pred    = torch.max(model_pred,1)
            n_correct   += (y_pred==y_trgt).sum().item()
            #print(y_trgt)
            n_total     += batch_in.size(0)
            
            y_probs += list(y_prob.cpu().numpy())
            
        val_accr  = (n_correct/n_total)
        top2_pi      = (top2_pi_sum/n_total).detach().cpu().item()
        entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'val_accr':val_accr,'epis':epis,'alea':alea, 'top2_pi':top2_pi,
                    'pi_entropy':entropy_pi_avg}
        model.train() # back to train mode 
    return out_eval

def gather_uncertainty_sdn(model,data_iter,data_size,device):
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum = 0,0,0,0
        epis_ = list()
        alea_ = list()
        pi_entropy_ = list()
        maxsoftmax_ = list()
        entropy_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            y_trgt      = batch_out.to(device)
            mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [B x N]

            #print(pi)
            # Compute uncertainty 
            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]
            pi_entropy  = unct_out['pi_entropy']
            
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            
            y_prob,y_pred = torch.max(model_pred,1)
            
            entropy = -torch.sum(model_pred*torch.log(model_pred),dim=-1)

            n_correct   += (y_pred==y_trgt).sum().item()
            n_total     += batch_in.size(0)
            maxsoftmax_ += list(1-y_prob.cpu().numpy())
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())
            entropy_    += list(entropy.cpu().numpy())

        val_accr  = (n_correct/n_total)
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'val_accr':val_accr,'epis':epis,'alea':alea, 
                        'epis_' : epis_,'alea_' : alea_, 'maxsoftmax_':maxsoftmax_,
                        'pi_entropy_':pi_entropy_,'entropy_':entropy_}
    return out_eval
    
def mln_transitionmatrix(model,data_iter,data_size,device,num,label=10):
    #false_label=torch.tensor([10]).to(device)
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        D1=np.zeros((label,label))
        D2=np.zeros((label,label))
        D3=np.zeros((label,label))
        confusion_matrix = np.zeros((label,label))
        sigma_out={}
        for i in range(label):
            sigma_out[str(i)]=list()
        for batch_in,batch_out in data_iter:
            # Foraward path
            if data_size is None:
                mln_out     = model.forward(batch_in.to(device))
            else:
                mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out = mln_eval(pi,mu,sigma,num+3,label)
            mu_prime=out['mu_prime']
            mu_sel1=out['mu_sel'] # [N x D]
            mu_sel2=out['mu_sel2']
            unct_out = mln_uncertainties(pi,mu,sigma)
            _,y = torch.max(mu_sel1,dim=-1)
            for i in range(batch_in.size(0)):
                D1[y[i].item()] +=mu_sel1[i].cpu().numpy()
                D2[y[i].item()] +=mu_sel2[i].cpu().numpy()
                D3[y[i].item()] +=mu_prime[i].cpu().numpy()
                sigma_out[str(y[i].item())].append(unct_out['alea'][i].cpu().numpy().tolist())
                confusion_matrix[y[i].item(),batch_out[i].item()]+=1
        model.train() # back to train mode 
        out_eval = {'D1':D1,'D2':D2,'D3':D3,
                    'sigma':sigma_out,'confusion_matrix':confusion_matrix}
    return out_eval

def get_th(clean_eval,ambiguous_eval):
    list = ambiguous_eval['alea_']+clean_eval['alea_']
    #true = [1]*len(ambiguous_eval['alea_'])+[0]*len(clean_eval['alea_'])
    list_=np.asarray(list)
    res=np.median(list_)
    alea1=np.asarray(ambiguous_eval['alea_'])
    alea2=np.asarray(clean_eval['alea_'])
    indices_amb1 = np.where(alea1<res)[0]
    indices_clean1=np.where(alea2<res)[0]
    indices_amb2 = np.where(alea1>res)[0]
    indices_clean2=np.where(alea2>res)[0]
    print(indices_amb1.shape,indices_clean1.shape)
    return indices_amb1,indices_clean1,indices_amb2,indices_clean2