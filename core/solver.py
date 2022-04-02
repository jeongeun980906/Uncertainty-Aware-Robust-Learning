import torch
from core.loader import build_model,build_dataset
from core.plot import *
from core.loss import mace_loss, mce_loss
from core.eval import func_eval, mln_transitionmatrix, gather_uncertainty_sdn,get_th
from dataloader.dirty_estimate import get_estimated_dataset
from core.mixup import mixup_data
import torch.optim as optim
import os

def load(args):
    args=args
    device='cuda'
    train_iter,val_iter,test_iter,dataset_config,trec_config=build_dataset(args)
    net=build_model(args,device,trec_config)
    config={
            "k":args.k,
            "sig_min":args.sig_min, 'sig_max': args.sig_max,
            "lr":args.lr,"wd":args.wd,'lr_rate':args.lr_rate,
            'ratio1':args.ratio,"ratio2":args.ratio2, 'lr_step':args.lr_step               
        }
    return train_iter,val_iter,test_iter,net,config,dataset_config

def train(args,train_iter,val_iter,test_iter,MLN,config,dataset_config):
    labels=int(dataset_config['num_classes'])
    transition_matrix = dataset_config['transition_matrix']
    device='cuda'
    data_size=dataset_config['input_size']
    ratio1=config['ratio1']
    ratio2=config['ratio2']

    DIR = './res/'+str(args.data)+'_'+str(args.mode)+'_'+str(args.ER)+'/'
    cDIR = './ckpt/{}_{}_{}/'.format(args.data,args.mode,args.ER)
    txtName = (DIR+str(args.id)+'_log.txt')
    try:
        print('dir made')
        os.makedirs(DIR)
        os.makedirs(cDIR)
    except FileExistsError:
        pass

    f = open(txtName,'w') # Open txt file
    print_n_txt(_f=f,_chars='Text name: '+txtName)
    print_n_txt(_f=f,_chars=str(args))
    if dataset_config['val_noise_rate'] != None:
        strTemp = ('cross validation set actual noise rate: %.3f'%(dataset_config['val_noise_rate']))
        print_n_txt(_f=f,_chars=strTemp)
    if args.data=='trec':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, MLN.parameters()), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_rate, step_size=10)
    elif args.data=='mnist' or args.data == 'dirty_mnist':
        # optimizer = optim.SGD(MLN.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.wd)
        optimizer = optim.Adam(MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config['lr_rate'], step_size=config['lr_step'])
    elif args.data == 'clothing1m':
        # optimizer = optim.Adam(MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        optimizer= optim.SGD(MLN.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config['lr_rate'], step_size=config['lr_step'])
    else:
        # if args.resnet or args.mixup:
        #     print('SGD')
        #     optimizer = optim.SGD(MLN.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.wd)
        # else:
        # optimizer = optim.SGD(MLN.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.wd)
        optimizer = optim.Adam(MLN.parameters(),lr=args.lr,weight_decay=args.wd,eps=1e-8)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config['lr_rate'], step_size=config['lr_step'])
    MLN.train()
    EPOCHS = args.epoch
    train_acc, test_acc = [], []
    for epoch in range(EPOCHS):
        loss_sum = 0.0
        #time.sleep(1)
        for e, (batch_in,batch_out) in enumerate(train_iter):
            # print(e)
            if e%100==0:
                print(e)
            if args.mixup:
                batch_in, y1, y2, lam = mixup_data(batch_in.to(device),batch_out.to(device)
                                                    ,alpha=args.alpha)
                mln_out = MLN.forward(batch_in.view(data_size).to(device))

                target1 = torch.eye(labels)[y1].to(device)
                target1=target1.to(device)

                target2 = torch.eye(labels)[y2].to(device)
                target2=target2.to(device)

                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                loss_out_1 = mace_loss(pi,mu,sigma,target1) # 'mace_avg','epis_avg','alea_avg'
                loss_1 = loss_out_1['mace_avg'] - ratio1*loss_out_1['epis_avg'] + ratio2*loss_out_1['alea_avg']

                loss_out_2 = mace_loss(pi,mu,sigma,target2) # 'mace_avg','epis_avg','alea_avg'
                loss_2 = loss_out_2['mace_avg'] - ratio1*loss_out_2['epis_avg'] + ratio2*loss_out_2['alea_avg']
                loss = lam*loss_1+(1-lam)*loss_2
                loss_out = {
                    'mace_avg': lam*loss_out_1['mace_avg'].item()+(1-lam)*loss_out_2['mace_avg'].item(),
                    'epis_avg': lam*loss_out_1['epis_avg'].item()+(1-lam)*loss_out_2['epis_avg'].item(),
                    'alea_avg': lam*loss_out_1['alea_avg'].item()+(1-lam)*loss_out_2['alea_avg'].item()
                }
                # print('flag3')
            else:
                # Forward path
                if data_size==None:
                    mln_out = MLN.forward(batch_in.to(device))
                else:
                    mln_out = MLN.forward(batch_in.view(data_size).to(device))
                    target = torch.eye(labels)[batch_out].to(device)
                    target=target.to(device)
            
                if args.sigma:
                    pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                    loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                    loss = loss_out['mace_avg'] - ratio1*loss_out['epis_avg'] + ratio2*loss_out['alea_avg']
                else:
                    # print(mln_out['sigma'])
                    pi, mu = mln_out['pi'], mln_out['mu']
                    loss_out = mce_loss(pi,mu,target)
                    loss = loss_out['mce_avg'] #- ratio1*loss_out['epis_avg']
                    
            # Weight Decay Loss
            # print('flag1')
            l2_reg = torch.tensor(0.)
            for param in MLN.parameters():
                l2_reg += torch.norm(param).cpu()
            #print(loss)
            # print('flag2')
            optimizer.zero_grad() # reset gradient
            loss.backward() # back-propagation
            optimizer.step() # optimizer update
            # Track losses
            loss_sum += loss
            # print('flag3')
        scheduler.step()
        loss_avg = loss_sum/len(train_iter)
        train_res = func_eval(MLN,train_iter,data_size,device,use_sigma=args.sigma)
        test_res  = func_eval(MLN,val_iter,data_size,device,use_sigma=args.sigma)
        if args.sigma:
            strTemp =  ("epoch:[%d/%d] loss:[%.4f] train_accr:[%.4f] Test_accr:[%.4f] \nmace_avg:[%.3f] epis avg:[%.3f] alea avg:[%.3f] weight decay loss: [%.3f]"%
                (epoch,EPOCHS,loss_avg,train_res['val_accr'],test_res['val_accr'],loss_out['mace_avg'],loss_out['epis_avg'],loss_out['alea_avg'],l2_reg.item()*args.wd))
        else:
            strTemp =  ("epoch:[%d/%d] loss:[%.4f] train_accr:[%.4f] Test_accr:[%.4f] \nmce_avg:[%.3f] epis avg:[%.3f] weight decay loss: [%.3f]"%
                (epoch,EPOCHS,loss_avg,train_res['val_accr'],test_res['val_accr'],loss_out['mce_avg'],loss_out['epis_avg'],l2_reg.item()*args.wd))

        strTemp2 = ("[Train] alea:[%.3f] epis:[%.3f] pi_entropy: [%.5f] top2_pi:[%.3f]\n[Test] alea:[%.3f] epis:[%.5f] pi_entropy:[%.5f] top2_pi:[%.3f]"%
                    (train_res['alea'],train_res['epis'],train_res['pi_entropy'], train_res['top2_pi'],\
                    test_res['alea'],test_res['epis'],test_res['pi_entropy'],test_res['top2_pi']))
        print_n_txt(_f=f,_chars=strTemp)
        print_n_txt(_f=f,_chars=strTemp2)          
        train_acc.append(train_res['val_accr'])
        test_acc.append(test_res['val_accr'])
    torch.save(MLN.state_dict(),'./ckpt/{}_{}_{}/MLN_{}.pt'.format(args.data,args.mode,args.ER,args.id))

    save_log_dict(args,train_acc,test_acc)
    plot_res_once(train_acc,test_acc,args,DIR)
    if args.mode != 'instance':
        if len(test_iter)==1:
            print(ratio1,ratio2)
            out = mln_transitionmatrix(MLN,test_iter[0],data_size,device
                                ,dataset_config["num"],labels,use_sigma=args.sigma)
            plot_tm_ccn(out,args,transition_matrix,labels)
            var=avg_total_variance(out['D3'],transition_matrix)
            rank=kendall_tau(out['D3'],transition_matrix)
            strtemp=('avarage total variance: [%.4f] kendalltau: [%.4f]'%(var,rank))
            print_n_txt(_f=f,_chars=strtemp)

        else:
            N=dataset_config["num"]
            clean_eval = gather_uncertainty_sdn(MLN,test_iter[1],data_size,device)
            ambiguous_eval=gather_uncertainty_sdn(MLN,test_iter[0],data_size,device)
            auroc = plot_hist(clean_eval,ambiguous_eval,args)
            strtemp=('auroc_alea: [%.4f] auroc_epis: [%.4f] auroc_pi_entropy: [%.4f] auroc_maxsoftmax: [%.4f] auroc_entropy: [%.4f]'%
                        (auroc['alea_'],auroc['epis_'],auroc['pi_entropy_'],auroc['maxsoftmax_'],auroc['entropy_']))
            print_n_txt(_f=f,_chars=strtemp)
            indices_amb1,indices_clean1,indices_amb2,indices_clean2=get_th(clean_eval,ambiguous_eval)
            del test_iter
            e_amb_iter,e_clean_iter = get_estimated_dataset(indices_amb1,indices_clean1,indices_amb2,indices_clean2,args)
            out1=mln_transitionmatrix(MLN,e_clean_iter,data_size,'cuda',N)
            out2=mln_transitionmatrix(MLN,e_amb_iter,data_size,'cuda',N)
            
            plot_tm_sdn(out1,out2,transition_matrix,args)
            plot_alea_sdn(out1,out2,args)

            var=avg_total_variance(out1['D3'],np.eye(labels))
            rank=kendall_tau(out1['D3'],np.eye(labels))
            strtemp = ('avarage total variance_clean: {} kendalltau_clean {}'.format(var,rank))
            print_n_txt(_f=f,_chars=strtemp)

            var=avg_total_variance(out2['D3'],transition_matrix)
            rank=kendall_tau(out2['D3'],transition_matrix)
            strtemp = ('avarage total variance_ambiguous: {} kendalltau_ambiguous {}'.format(var,rank))
            print_n_txt(_f=f,_chars=strtemp)


def test(args,test_iter,MLN,config,dataset_config):
    labels=int(dataset_config['num_classes'])
    transition_matrix = dataset_config['transition_matrix']
    device='cuda'
    data_size=dataset_config['input_size']
    ratio1=config['ratio1']
    ratio2=config['ratio2']
    state_dict=torch.load('./ckpt/{}_{}_{}/MLN_{}.pt'.format(args.data,args.mode,args.ER,args.id))
    MLN.load_state_dict(state_dict)
    if len(test_iter)==1:
        out = mln_transitionmatrix(MLN,test_iter[0],data_size,device,dataset_config["num"],labels)
        plot_tm_ccn(out,args,transition_matrix,labels,ratio1,ratio2)
        var=avg_total_variance(out['D3'],transition_matrix)
        rank=kendall_tau(out['D3'],transition_matrix)
        print('avarage total variance: {} kendalltau: {}'.format(var,rank))
    else:
        N=dataset_config["num"]
        clean_eval = gather_uncertainty_sdn(MLN,test_iter[1],data_size,device)
        ambiguous_eval=gather_uncertainty_sdn(MLN,test_iter[0],data_size,device)
        print(ambiguous_eval['alea'],clean_eval['alea'])
        auroc = plot_hist(clean_eval,ambiguous_eval,args)
        print('auroc_alea: [%.4f] auroc_epis: [%.4f] auroc_pi_entropy: [%.4f] auroc_maxsoftmax: [%.4f] auroc_entropy: [%.4f]'%
                        (auroc['alea_'],auroc['epis_'],auroc['pi_entropy_'],auroc['maxsoftmax_'],auroc['entropy_']))
        indices_amb1,indices_clean1,indices_amb2,indices_clean2=get_th(clean_eval,ambiguous_eval)
        e_amb_iter,e_clean_iter = get_estimated_dataset(indices_amb1,indices_clean1,indices_amb2,indices_clean2,args)
        out1=mln_transitionmatrix(MLN,e_clean_iter,data_size,'cuda',N)
        out2=mln_transitionmatrix(MLN,e_amb_iter,data_size,'cuda',N)
        plot_tm_sdn(out1,out2,transition_matrix,args)
        plot_alea_sdn(out1,out2,args)


