import torch
import wandb
from core.train import build_model,build_dataset
from core.plot_MLN import *
from core.MLN_loss import mace_loss
from core.MLN_eval import func_eval,test_eval,func_eval2
import torch.optim as optim
import os
from core.tunner import lambda_tunner
def load(args):
    args=args
    device='cuda'
    train_iter,val_iter,test_iter,dataset_config,trec_config=build_dataset(args)
    net=build_model(args,device,trec_config)
    config_defalts={
            "k":args.k,
            "sig_min":args.sig_min, 'sig_max': args.sig_max,
            "lr":args.lr,"wd":args.wd,'lr_rate':args.lr_rate
        }
    if args.sweep and args.train:
        wandb.init(config=config_defalts)
        wandb.watch(net, log_freq=100)
        config=wandb.config
        print(config)
    elif args.wandb and args.train:
        wandb.init(config=config_defalts,
            tags=['{}_{}_{}_{}'.format(args.data,args.mode,args.ER,args.id)],
            name=str(args.id), # Run name
            project='MLN_{}_{}_{}'.format(args.data,args.mode,args.ER)# Project name. Default: gnn-robot
        )
        wandb.config.update({'ratio1':args.ratio,"ratio2":args.ratio2})
        config=wandb.config
        print(config)
    else:
        config={
            "k":args.k,
            "sig_min":args.sig_min, 'sig_max': args.sig_max,
            "lr":args.lr,"wd":args.wd,'lr_rate':args.lr_rate,
            'ratio1':args.ratio,"ratio2":args.ratio2                
        }
    return train_iter,val_iter,test_iter,net,config,dataset_config

def train(args,train_iter,val_iter,test_iter,MLN,config,dataset_config):
    labels=int(dataset_config['num_classes'])
    transition_matrix = dataset_config['transition_matrix']
    device='cuda'
    data_size=dataset_config['input_size']
    ratio1=config['ratio1']
    ratio2=config['ratio2']

    if args.sweep:
        DIR='./res/sweep/{}_{}_{}/'.format(args.data,args.mode,args.ER)
        cDIR = './ckpt/sweep/{}_{}_{}/'.format(args.data,args.mode,args.ER)
        txtName=(DIR+'{}_{}_{}_log.txt'.format(args.id,ratio1,ratio2))
    else:
        DIR = './res/normal/'+str(args.data)+'_'+str(args.mode)+'_'+str(args.ER)+'/'
        cDIR = './ckpt/normal/{}_{}_{}/'.format(args.data,args.mode,args.ER)
        txtName = (DIR+str(args.id)+'_log.txt')
    try:
        print('dir made')
        os.mkdir(DIR)
        os.mkdir(cDIR)
    except FileExistsError:
        pass

    if args.tunner:
        print('use lambda tunner')
        tunner = lambda_tunner(int(dataset_config['num_classes']))
        ratio1=tunner[args.data][0]
        ratio2 = tunner[args.data][1]
    f = open(txtName,'w') # Open txt file
    print_n_txt(_f=f,_chars='Text name: '+txtName)
    print_n_txt(_f=f,_chars=str(args))
    if args.data=='trec':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_rate, step_size=1000)
    elif args.data=='mnist':
        optimizer = optim.Adam(MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,9,12,15,18], gamma=args.lr_rate)
    else:
        optimizer = optim.Adam(MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
    
    MLN.train()
    EPOCHS = args.epoch
    train_acc, test_acc = [], []
    tunner_rate=20
    for epoch in range(EPOCHS):
        loss_sum = 0.0
        #time.sleep(1)
        for batch_in,batch_out in train_iter:
            # Forward path
            if data_size==None:
                mln_out = MLN.forward(batch_in.to(device))
            else:
                mln_out = MLN.forward(batch_in.view(data_size).to(device))
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
        loss_avg = loss_sum/len(train_iter)
        train_res = func_eval(MLN,train_iter,data_size,device)
        test_res  = func_eval(MLN,val_iter,data_size,device)
        strTemp =  ("epoch:[%d/%d] loss:[%.4f] train_accr:[%.4f] Test_accr:[%.4f] \nmace_avg:[%.3f] epis avg:[%.3f] alea avg:[%.3f]"%
                (epoch,EPOCHS,loss_avg,train_res['val_accr'],test_res['val_accr'],loss_out['mace_avg'],loss_out['epis_avg'],loss_out['alea_avg']))
        strTemp2 = ("[Train] alea:[%.3f] epis:[%.3f] pi_entropy: [%.5f] top2_pi:[%.3f]\n[Test] alea:[%.3f] epis:[%.5f] pi_entropy:[%.5f] top2_pi:[%.3f]"%
                    (train_res['alea'],train_res['epis'],train_res['pi_entropy'], train_res['top2_pi'],\
                    test_res['alea'],test_res['epis'],test_res['pi_entropy'],test_res['top2_pi']))
        print_n_txt(_f=f,_chars=strTemp)
        print_n_txt(_f=f,_chars=strTemp2)          
        if args.sweep or args.wandb:
            wandb.log({"loss": loss_avg,
                        "train_acc": train_res['val_accr'],
                        "test_acc": test_res['val_accr'],
                        'alea': test_res['alea'],
                        'epis': test_res['epis'],
                        'pi_entropy': test_res['pi_entropy'],
                        'top2_pi': test_res['top2_pi'],
            })
        train_acc.append(train_res['val_accr'])
        test_acc.append(test_res['val_accr'])
    if args.sweep:
        torch.save(MLN.state_dict(),'./ckpt/sweep/{}_{}_{}/MLN_{}_{:.1f}_{:.1f}.pt'.format(args.data,args.mode,args.ER,args.id,ratio1,ratio2))
    else:
        torch.save(MLN.state_dict(),'./ckpt/normal/{}_{}_{}/MLN_{}.pt'.format(args.data,args.mode,args.ER,args.id))

    save_log_dict(args,train_acc,test_acc)
    plot_res_once(train_acc,test_acc,args,DIR)
    try:
        if len(test_iter)==1:
            print(ratio1,ratio2)
            out = test_eval(MLN,test_iter[0],data_size,device,dataset_config["num"],labels)
            plot_pi(out['pi1'],out['pi2'],args,labels,ratio1,ratio2)
            plot_mu(out,args,transition_matrix,labels,ratio1,ratio2)
            var=avg_total_variance(out['D3'],transition_matrix)
            rank=kendall_tau(out['D3'],transition_matrix)
            strtemp=('avarage total variance: {} kendalltau: {}'.format(var,rank))
            print_n_txt(_f=f,chars=strtemp)

        else:
            N=dataset_config["num"]
            clean_eval = func_eval2(MLN,test_iter[1],'cuda')
            ambiguous_eval=func_eval2(MLN,test_iter[0],'cuda')
            plot_hist(clean_eval,ambiguous_eval,args)

            out1=test_eval(MLN,test_iter[1],data_size,'cuda',N)
            out2=test_eval(MLN,test_iter[0],data_size,'cuda',N)
            plot_pi(out1['pi1'],out1['pi2'],out2['pi1'],out2['pi2'],args,N)
            plot_mu(out1,out2,transition_matrix,args)
            plot_sigma2(out1,out2,args)
    except:
        print('error')
        pass
def test(args,train_iter,val_iter,test_iter,MLN,config,dataset_config):
    labels=int(dataset_config['num_classes'])
    transition_matrix = dataset_config['transition_matrix']
    device='cuda'
    data_size=dataset_config['input_size']
    ratio1=config['ratio1']
    ratio2=config['ratio2']
    if args.sweep:
        state_dict=torch.load('./ckpt/sweep/{}_{}_{}/MLN_{}_{:.1f}_{:.1f}.pt'.format(args.data,args.mode,args.ER,args.id,args.ratio,args.ratio2))
    else:
        state_dict=torch.load('./ckpt/normal/{}_{}_{}/MLN_{}.pt'.format(args.data,args.mode,args.ER,args.id))
    MLN.load_state_dict(state_dict)
    if len(test_iter)==1:
        out = test_eval(MLN,test_iter[0],data_size,device,dataset_config["num"],labels)
        plot_pi(out['pi1'],out['pi2'],args,labels,ratio1,ratio2)
        plot_mu(out,args,transition_matrix,labels,ratio1,ratio2)
        var=avg_total_variance(out['D3'],transition_matrix)
        rank=kendall_tau(out['D3'],transition_matrix)
        print('avarage total variance: {} kendalltau: {}'.format(var,rank))
    else:
        N=dataset_config["num"]
        clean_eval = func_eval2(MLN,test_iter[1],'cuda')
        ambiguous_eval=func_eval2(MLN,test_iter[0],'cuda')
        plot_hist(clean_eval,ambiguous_eval,args)
        
        out1=test_eval(MLN,test_iter[1],data_size,'cuda',N)
        out2=test_eval(MLN,test_iter[0],data_size,'cuda',N)
        plot_pi(out1['pi1'],out1['pi2'],out2['pi1'],out2['pi2'],args,N)
        plot_mu(out1,out2,transition_matrix,args)
        plot_sigma2(out1,out2,args)