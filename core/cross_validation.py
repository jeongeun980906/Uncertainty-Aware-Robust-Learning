import torch
from core.plot import *
from core.loss import mace_loss,mln_gather
from core.solver import load,train
from core.loader import get_cross_val_dataset
import torch.optim as optim
import os

class cross_validation():
    def __init__(self,args):
        self.args= args
        self.device='cuda'

    def load_total_dataset(self):
        self.train_iter,_,self.test_iter,self.MLN,self.config,self.dataset_config = load(self.args)
    
    def load_new_dataset(self):
        del self.train_iter
        self.train_iter,self.val_iter,val_noise_rate = get_cross_val_dataset(self.args,self.val_indicies)
        self.dataset_config['val_noise_rate'] = val_noise_rate
    def train_full(self):
        self.MLN.init_param()
        train(self.args,self.train_iter,self.val_iter,self.test_iter,self.MLN,self.config,self.dataset_config)

    def small_loss(self):
        labels=int(self.dataset_config['num_classes'])
        res = []
        for c in range(labels):
            a = torch.where(self.mace[:,1]==c)[0]
            values = self.mace[a,0]
            indicies = self.mace[a,2]
            _,value = torch.topk(values,int(0.1*self.train_size/labels)
                                ,largest=False)
            res.extend(indicies[value].numpy().tolist())
        return res
    
    def gain_traisiton_matrix(self):
        self.MLN.init_param()
        data_size=self.dataset_config['input_size']
        ratio1=self.config['ratio1']
        ratio2=self.config['ratio2']
        labels=int(self.dataset_config['num_classes'])
        device = self.device
        args=self.args
        if args.data=='trec':
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.MLN.parameters()), lr=args.lr, weight_decay=args.wd)
            #optimizer = optim.Adam(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.data=='mnist':
            optimizer = optim.Adam(self.MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,9,12,15,18], gamma=0.2)
        else:
            optimizer = optim.Adam(self.MLN.parameters(),lr=args.lr,weight_decay=self.config['wd'],eps=1e-8)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.2, step_size=10)
        self.MLN.train()
        EPOCHS = args.epoch
        for epoch in range(EPOCHS):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in self.val_iter:
                # print(batch_out)
                # Forward path
                if data_size==None:
                    mln_out = self.MLN.forward(batch_in.to(device))
                else:
                    mln_out = self.MLN.forward(batch_in.view(data_size).to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(labels)[batch_out].to(device)
                target=target.to(device)
                loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                loss = loss_out['mace_avg'] - ratio1*loss_out['epis_avg'] + ratio2*loss_out['alea_avg']
                #print(loss)
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            scheduler.step()
            print("Average Loss: %.3f"%(loss_sum/len(self.train_iter)))
        self.MLN.eval()
        self.mace = list()
        transition_matrix = np.zeros((labels,labels))
        with torch.no_grad():
            for batch_in,batch_out in self.train_iter:
                if data_size==None:
                    mln_out = self.MLN.forward(batch_in.to(device))
                else:
                    mln_out = self.MLN.forward(batch_in.view(data_size).to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                model_pred = mln_gather(pi,mu,sigma)['mu_sel']
                _,y_pred    = torch.max(model_pred,1) # [N]
                for a,i in enumerate(y_pred.cpu().numpy().tolist()):
                    transition_matrix[i,batch_out[a].item()] += 1
        with torch.no_grad():
            for batch_in,batch_out in self.val_iter:
                if data_size==None:
                    mln_out = self.MLN.forward(batch_in.to(device))
                else:
                    mln_out = self.MLN.forward(batch_in.view(data_size).to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                model_pred = mln_gather(pi,mu,sigma)['mu_sel']
                _,y_pred    = torch.max(model_pred,1) # [N]
                acc = 0
                tot = 0
                for a,i in enumerate(y_pred.cpu().numpy().tolist()):
                    transition_matrix[i,batch_out[a].item()] += 1
                    tot +=1
                    if i == batch_out[a].item():
                        acc+=1
        print('VAL ACC: {}'.format(acc/tot))
        for i in range(labels):
            transition_matrix[i,:]=transition_matrix[i,:]/(np.sum(transition_matrix[i,:]))
        transition_matrix = np.nan_to_num(transition_matrix)
        self.MLN.train()
        self.dataset_config['transition_matrix'] = transition_matrix#.tolist()
        print(self.dataset_config)

    def train_init(self):
        data_size=self.dataset_config['input_size']
        ratio1=self.config['ratio1']
        ratio2=self.config['ratio2']
        labels=int(self.dataset_config['num_classes'])
        device = self.device
        args=self.args
        if args.data=='trec':
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.MLN.parameters()), lr=args.lr, weight_decay=args.wd)
            #optimizer = optim.Adam(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.data=='mnist':
            optimizer = optim.Adam(self.MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        else:
            optimizer = optim.Adam(self.MLN.parameters(),lr=args.lr,weight_decay=self.config['wd'],eps=1e-8)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        self.MLN.train()
        if args.data == 'mnist':
            EPOCHS = 5 
        elif args.data == 'cifar100':
            EPOCHS = 50
        else:
            EPOCHS = 20
        train_acc, test_acc = [], []
        for epoch in range(EPOCHS):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in self.train_iter:
                # Forward path
                if data_size==None:
                    mln_out = self.MLN.forward(batch_in.to(device))
                else:
                    mln_out = self.MLN.forward(batch_in.view(data_size).to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(labels)[batch_out].to(device)
                target=target.to(device)
                loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                loss = loss_out['mace_avg'] - ratio1*loss_out['epis_avg'] + ratio2*loss_out['alea_avg']
                #print(loss)
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            print("Average Loss: %.3f"%(loss_sum/len(self.train_iter)))
        self.MLN.eval()
        self.mace = list()
        indx = 0
        with torch.no_grad():
            for batch_in,batch_out in self.train_iter:
                if data_size==None:
                    mln_out = self.MLN.forward(batch_in.to(device))
                else:
                    mln_out = self.MLN.forward(batch_in.view(data_size).to(device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(labels)[batch_out].to(device)
                target = target.to(device)
                loss_out = mace_loss(pi,mu,sigma,target)
                mace = loss_out['mace'].cpu().numpy().tolist()
                for i,m in enumerate(mace):
                    self.mace.append([m,batch_out[i].item(),indx])
                    indx += 1
        self.train_size = indx
        self.mace = torch.tensor(self.mace)
        print(self.mace.size())
        self.MLN.train()
        self.val_indicies = self.small_loss()