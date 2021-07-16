from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import numpy as np
import os
import json
from scipy.stats import kendalltau

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def avg_total_variance(wm,tm):
    labels=wm.shape[0]
    temp=np.sum(abs(wm-tm),axis=1)
    return 1/labels*np.sum(temp/2)

def kendall_tau(wm,tm):
    labels=wm.shape[0]
    res=0
    for i in range(labels):
        tau, p_value=kendalltau(wm[i,:],tm[i,:])
        res+=tau
    return res/labels


def plot_res_once(train_acc,test_acc,args,DIR):
    if args.mode=='rp':
        errType ='Permutattion'
    elif args.mode=='rs':
        errType ='Random Shuffle'
    else:
        errType=args.mode
    errRate=args.ER
    save_dir=DIR+str(args.id)+'_train_res.png'
    plt.figure(figsize=(7,5))
    chTr,=plt.plot(train_acc,label='MLN (train)',color='k',lw=2,ls='--',marker='')
    chTe,=plt.plot(test_acc,label='MLN (test)',color='b',lw=2,ls='-',marker='')
    plt.legend(handles=[chTr,chTe],loc='lower center',shadow=True,ncol=2,fontsize=10.5)
    plt.xlabel('Number of Epochs',fontsize=15)
    plt.ylabel('Accuracy (%)',fontsize=13)
    plt.title('%d%% %s'%(errRate*100,errType),fontsize=13)
    plt.savefig(save_dir)
    print ("%s saved."%(save_dir))
    

def plot_res(train_acc,test_acc,legend,name):
    save_dir='./res/{}_train_res.png'.format(name)
    plt.figure(figsize=(14,10))
    plt.suptitle('{} Accuracy'.format(name),fontsize=15)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title("{} {} {}".format(name,legend[i][:-5],legend[i][-4:-1]))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(train_acc[i],label='train',color='k',lw=2,ls='--',marker='')
        plt.plot(test_acc[i],label='test',color='b',lw=2,ls='-',marker='')
        plt.legend(loc=4,fontsize='medium') 
    plt.tight_layout()
    
    plt.savefig(save_dir)

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

def save_log_dict(args,train_acc,test_acc):
    if args.sweep:
        DIR_PATH='./res/log/sweep_{}_{}_{}_{}.json'.format(args.data,args.mode,args.ER,args.id)
    else:
        DIR_PATH='./res/log/'+str(args.data)+'_'+str(args.mode)+'_'+str(args.ER)+'_'+str(args.id)+'.json'
    try:
        with open(DIR_PATH,'r') as json_file:
            data=json.load(json_file)
            print(data)
    except:
        data={}
    with open(DIR_PATH,'w') as json_file:
        dict_name1='train'
        dict_name2='test'
        data[dict_name1]=train_acc
        data[dict_name2]=test_acc
        print(data)
        json.dump(data,json_file, indent=4)

def plot_pi(out,out2,args,labels=10,ratio1=1,ratio2=1):
    if args.sweep:
        DIR='./res/rate_res/{}_{}_{}/sweep_{}_{}_{}_test_pi.png'.format(args.data,args.mode,args.ER,args.id,ratio1,ratio2)
        DIR2='./res/rate_res/{}_{}_{}/sweep_{}_{}_{}_test_pi.json'.format(args.data,args.mode,args.ER,args.id,ratio1,ratio2)
    else:  
        DIR='./res/rate_res/{}_{}_{}/{}_test_pi.png'.format(args.data,args.mode,args.ER,args.id)
        DIR2='./res/rate_res/{}_{}_{}/{}_test_pi.json'.format(args.data,args.mode,args.ER,args.id)
    if args.mode not in ['symmetric','rp','rs','fairflip','mixup']:
        if args.mode=='asymmetric':
            n=3
        else:
            n=int(args.mode[-1])+2
        if args.data=='cifar10':
            IS_NOISE=[2,3,4,5,9]
            NOT_NOISE=[0,1,6,7,8]
        elif args.data=='mnist':
            IS_NOISE = [7,2,5,6,3]
            NOT_NOISE=[0,1,4,8,9]
        elif args.data=='trec':
            IS_NOISE = [1,2,3]
            NOT_NOISE=[0,4,5]
        else:
            IS_NOISE=[i for i in range(labels)]
            NOT_NOISE=[]
    else:
        n=3
        IS_NOISE=[i for i in range(labels)]
        NOT_NOISE=[]
    log = []
    log2 = []
    x=[]
    for i in range(labels):
        try:
            pi=out[str(i)]
            dist=np.asarray(pi).T
            mean_dist=np.mean(dist,axis=1)
            log.append(mean_dist)
            pi=out2[str(i)]
            dist=np.asarray(pi).T
            mean_dist=np.mean(dist,axis=1)
            log2.append(mean_dist)
            x.append(i)
        except:
            print(i)
            pass
    log=np.asarray(log)
    log2=np.asarray(log2)
    log2=np.transpose(log2)
    log=np.transpose(log)
    fig = plt.figure(figsize=(9,7))
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    plt.suptitle('{} {} {}\% Mixture Weight'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)
    plt.subplot(2, 1, 1) 
    plt.title('Delta Mixture Weight',fontsize=18)
    plt.xlabel('Class',fontsize=15)
    if args.data != 'cifar100':
        # if args.mode=='symmetric':
        #     plt.ylim([0,0.3])
        plt.xticks([i for i in range(labels)])
        clean=np.take(log,NOT_NOISE,axis=1)
        noisy=np.take(log,IS_NOISE,axis=1)
        for i in range(n):
            plt.plot(log[i,:],label='$\pi({})-\pi({})$'.format(i+1,i+2),linewidth=1)
        plt.plot(NOT_NOISE,clean[0,:],'bo',markersize=5,label='clean label')
        plt.plot(IS_NOISE,noisy[0,:],'ro',markersize=5,label='noisy label')
        if args.mode not in ['symmetric','asymmetric','fairflip']:
            plt.plot(NOT_NOISE,clean[n-2,:],'bo',markersize=5)
            plt.plot(IS_NOISE,noisy[n-2,:],'ro',markersize=5)
        else:
            plt.plot(NOT_NOISE,clean[1,:],'bo',markersize=5)
            plt.plot(IS_NOISE,noisy[1,:],'ro',markersize=5)
        plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='x-large')
        plt.tight_layout()
    else:
        for i in range(n):
            plt.plot(x,log[i,:],label='$\pi({})-\pi({})$'.format(i+1,i+2),linewidth=1)
        plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='x-large')
        plt.tight_layout()        
    plt.subplot(2, 1, 2) 
    # print(log)
    plt.title('Mixture Weight',fontsize=18)
    plt.xlabel('Class',fontsize=15)
    if args.data !='cifar100':
        plt.xticks([i for i in range(labels)])
        clean2=np.take(log2,NOT_NOISE,axis=1)
        noisy2=np.take(log2,IS_NOISE,axis=1)
        for i in range(n+1):
            plt.plot(log2[i,:],label='$\pi({})$'.format(i+1),linewidth=1)
        plt.plot(NOT_NOISE,clean2[0,:],'bo',markersize=5,label='clean label')
        plt.plot(IS_NOISE,noisy2[0,:],'ro',markersize=5,label='noisy label')
        if args.mode not in ['symmetric','asymmetric','fairflip']:
            for i in range(1,n-1):
                plt.plot(NOT_NOISE,clean2[i,:],'bo',markersize=5)
                plt.plot(IS_NOISE,noisy2[i,:],'ro',markersize=5)
        else:
            plt.plot(NOT_NOISE,clean2[1,:],'bo',markersize=5)
            plt.plot(IS_NOISE,noisy2[1,:],'ro',markersize=5)
        plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='x-large')
        plt.tight_layout()
    else:
        for i in range(n+1):
            plt.plot(x,log2[i,:],label='$\pi({})$'.format(i+1),linewidth=1)
        plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='x-large')
        plt.tight_layout()        
    try:
        print('dir made')
        os.mkdir('./res/rate_res/{}_{}_{}/'.format(args.data,args.mode,args.ER))
    except FileExistsError:
        pass
    plt.savefig(DIR)
    save_log={}
    save_log['pi']=log2.tolist()
    save_log['dpi']=log.tolist()
    with open(DIR2,'w') as json_file:
        json.dump(save_log,json_file)


def plot_mu(out,args,true_noise,labels=10,ratio1=1,ratio2=1):
    if args.sweep:
        DIR1='./res/rate_res/{}_{}_{}/sweep_{}_{}_{}_test_mu.png'.format(args.data,args.mode,args.ER,args.id,ratio1,ratio2)
        DIR2='./res/rate_res/{}_{}_{}/sweep_{}_{}_{}_confusion.png'.format(args.data,args.mode,args.ER,args.id,ratio1,ratio2)
    else:
        DIR1='./res/rate_res/{}_{}_{}/{}_test_mu.png'.format(args.data,args.mode,args.ER,args.id)
        DIR2='./res/rate_res/{}_{}_{}/{}_confusion.png'.format(args.data,args.mode,args.ER,args.id)
    img1=out['D1']
    img2=out['D2']
    img3=out['D3']
    img4= out['confusion_matrix']
    for i in range(labels):
        try:
            img1[i,:]=img1[i,:]/(np.sum(img1[i,:]))
            img2[i,:]=img2[i,:]/(np.sum(img2[i,:]))
            img3[i,:]=img3[i,:]/(np.sum(img3[i,:]))
            img4[i,:]=img4[i,:]/(np.sum(img4[i,:]))
        except:
            pass
    img1=np.round(img1,2)
    img2=np.round(img2,2)
    img3=np.round(img3,2)
    img4=np.round(img4,2)
    img7=np.round(true_noise,2)
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    if args.data=='cifar100':
        fig,(ax3,ax4) = plt.subplots(2,1, figsize=(12, 20))
        fig.suptitle('{} {} {}\% Transition Matrix'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=30)
        # ax1.set_title('1st largest distribution',fontsize=25)
        # img = sns.heatmap(img1,ax=ax1)
        # ax1.set_xlabel('predicted probablity',fontsize=20)
        # ax1.set_ylabel('predicted class',fontsize=20)
        
        # ax2.set_title('2nd largest distribution',fontsize=25)
        # img = sns.heatmap(img2,ax=ax2)
        # ax2.set_xlabel('predicted probablity',fontsize=20)
        # ax2.set_ylabel('predicted class',fontsize=20)
        ax3.set_title('Predicted',fontsize=25)
        img = sns.heatmap(img3,ax=ax3)
        ax3.set_xlabel('Clean Label',fontsize=20)
        ax3.set_ylabel('Noisy Label',fontsize=20)

        ax4.set_title('True',fontsize=25)
        img = sns.heatmap(img7,ax=ax4)
        ax4.set_xlabel('Clean Label',fontsize=20)
        ax4.set_ylabel('Noisy Label',fontsize=20)
        plt.tight_layout()
        fig.savefig(DIR1)

        # fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # fig.suptitle('{}_{}_{}'.format(args.data,args.mode,args.ER),fontsize=20)
        # ax1.set_title('Confusion matrix')
        # img = sns.heatmap(img4,ax=ax1)
        # ax1.set_xlabel('predicted class',fontsize=10)
        # ax1.set_ylabel('labels',fontsize=10)
        
        # ax2.set_title('True')
        # img = sns.heatmap(img7,ax=ax2)
    else:
        #fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 16))
        fig, (ax3,ax4) = plt.subplots(2, 1, figsize=(6, 12))
        fig.suptitle('{} {} {}\% Transition Matrix'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)
        # ax1.set_title('1st largest distribution',fontsize=20)
        # img = sns.heatmap(img1,ax=ax1,annot=True)
        # ax1.set_xlabel('predicted probablity',fontsize=13)
        # ax1.set_ylabel('predicted class',fontsize=13)
        
        # ax2.set_title('2nd largest distribution',fontsize=20)
        # img = sns.heatmap(img2,ax=ax2,annot=True)
        # ax2.set_xlabel('predicted probablity',fontsize=13)
        # ax2.set_ylabel('predicted class',fontsize=13)

        ax3.set_title('Predicted',fontsize=18)
        img = sns.heatmap(img3,ax=ax3,annot=True)
        ax3.set_xlabel('Clean Label',fontsize=15)
        ax3.set_ylabel('Noise Label',fontsize=15)

        ax4.set_title('True',fontsize=18)
        img = sns.heatmap(img7,ax=ax4,annot=True)
        ax4.set_xlabel('Clean Label',fontsize=15)
        ax4.set_ylabel('Noisy Label',fontsize=15)
        plt.tight_layout()
        fig.savefig(DIR1)

    #     fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 8))
    #     fig.suptitle('{}_{}_{}'.format(args.data,args.mode,args.ER),fontsize=20)
    #     ax1.set_title('Confusion matrix')
    #     img = sns.heatmap(img4,ax=ax1,annot=True)
    #     ax1.set_xlabel('predicted class',fontsize=10)
    #     ax1.set_ylabel('labels',fontsize=10)
        
    #     ax2.set_title('True')
    #     img = sns.heatmap(img7,ax=ax2,annot=True)
    # fig.savefig(DIR2)
    # print(img4)

def plot_hist(clean_eval,ambiguous_eval,args):
    plt.figure(figsize=(8,5))
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    plt.title('{} {} {}\% Aleatoric Uncertainty Histogram \n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=10)
    plt.hist(clean_eval['alea_'], color='b',label='clean',bins=100, alpha=0.5)
    plt.hist(ambiguous_eval['alea_'][:10000],color='r',label='ambiguous',bins=100, alpha=0.5)
    plt.legend()
    plt.savefig('./res/{}_{}/{}_hist.png'.format(args.mode,args.ER,args.id))
    roc=roc_auc_score(clean_eval,ambiguous_eval)
    return roc

def plot_pi2(out,out2,out3,out4,args,N):
    n=N+2
    DIR='./res/{}_{}/{}_pi.png'.format(args.mode,args.ER,args.id)
    if args.mode not in ['symmetric','rp','rs','fairflip','mixup']:
        IS_NOISE = [7,2,5,6,3]
        NOT_NOISE=[0,1,4,8,9]
    else:
        IS_NOISE=[i for i in range(10)]
        NOT_NOISE=[]
    log = np.zeros((10,n+1))
    log2 = np.zeros((10,n+1))
    log3 = np.zeros((10,n+1))
    log4 = np.zeros((10,n+1))
    for i in range(10):
        pi=out[str(i)]
        dist=np.asarray(pi).T
        try:
            mean_dist=np.mean(dist,axis=1)
            log[i]=mean_dist
        except:
            log[i]=np.zeros(n+1)
        pi=out2[str(i)]
        dist=np.asarray(pi).T
        try:
            mean_dist=np.mean(dist,axis=1)
            log2[i]=mean_dist
        except:
            log2[i]=np.zeros(n+1)
        pi=out3[str(i)]
        dist=np.asarray(pi).T
        try:
            mean_dist=np.mean(dist,axis=1)
            log3[i]=mean_dist
        except:
            log3[i]=np.zeros(n+1)
        pi=out4[str(i)]
        dist=np.asarray(pi).T
        try:
            mean_dist=np.mean(dist,axis=1)
            log4[i]=mean_dist
        except:
            log4[i]=np.zeros(n+1)
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    fig = plt.figure(figsize=(12,8))
    plt.suptitle('{} {} {}\% Mixture Weight\n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=18)
    plt.subplot(2, 2, 1) 
    plt.title('Delta Mixture Weight Clean',fontsize=15)
    plt.xticks([i for i in range(10)])
    plt.xlabel("Class",fontsize=12)
    clean=np.take(log,NOT_NOISE,axis=0)
    noisy=np.take(log,IS_NOISE,axis=0)
    for i in range(n):
        plt.plot(log[:,i],label='$\pi({})-\pi({})$'.format(i+1,i+2),linewidth=1)
    plt.plot(log[:,0],'bo',markersize=5,label='clean label')
    if args.mode not in ['symmetric','asymmetric','fairflip']:
        plt.plot(log[:,n-2],'bo',markersize=5)
        #plt.plot(IS_NOISE,noisy[:,n-2],'ro',markersize=5)
    else:
        plt.plot(log[:,1],'bo',markersize=5)
        #plt.plot(IS_NOISE,noisy[:,1],'ro',markersize=5)

    plt.subplot(2, 2, 2) 
    plt.title('Delta Mixture Weight Ambiguous',fontsize=15)
    plt.xticks([i for i in range(10)])
    plt.xlabel("Class",fontsize=12)
    clean=np.take(log3,NOT_NOISE,axis=0)
    noisy=np.take(log3,IS_NOISE,axis=0)
    for i in range(n):
        plt.plot(log3[:,i],label='$\pi({})-\pi({})$'.format(i+1,i+2),linewidth=1)
    plt.plot(NOT_NOISE,clean[:,0],'bo',markersize=5,label='clean label')
    plt.plot(IS_NOISE,noisy[:,0],'ro',markersize=5,label='noisy label')
    if args.mode not in ['symmetric','asymmetric','fairflip']:
        plt.plot(NOT_NOISE,clean[:,n-2],'bo',markersize=5)
        plt.plot(IS_NOISE,noisy[:,n-2],'ro',markersize=5)
    else:
        plt.plot(NOT_NOISE,clean[:,1],'bo',markersize=5)
        plt.plot(IS_NOISE,noisy[:,1],'ro',markersize=5)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.tight_layout()

    plt.subplot(2, 2, 3) 
    # print(log)
    plt.title('Mixture Weight Clean',fontsize=13)
    plt.xticks([i for i in range(10)])
    clean2=np.take(log2,NOT_NOISE,axis=0)
    noisy2=np.take(log2,IS_NOISE,axis=0)
    for i in range(n+1):
        plt.plot(log2[:,i],label='$\pi({})$'.format(i+1),linewidth=1)
    plt.plot(log2[:,0],'bo',markersize=5,label='clean label')
    #plt.plot(IS_NOISE,noisy2[:,0],'ro',markersize=5,label='noisy label')
    if args.mode not in ['symmetric','asymmetric','fairflip']:
        for i in range(1,n-1):
            plt.plot(log2[:,i],'bo',markersize=5)
            #plt.plot(IS_NOISE,noisy2[:,i],'ro',markersize=5)
    else:
        plt.plot(log2[:,1],'bo',markersize=5)
        #plt.plot(IS_NOISE,noisy2[:,1],'ro',markersize=5)

    plt.subplot(2, 2, 4) 
    # print(log)
    plt.title('Mixture Weight Ambiguous',fontsize=13)
    plt.xticks([i for i in range(10)])
    clean2=np.take(log4,NOT_NOISE,axis=0)
    noisy2=np.take(log4,IS_NOISE,axis=0)
    for i in range(n+1):
        plt.plot(log4[:,i],label='$\pi({})$'.format(i+1),linewidth=1)
    plt.plot(NOT_NOISE,clean2[:,0],'bo',markersize=5,label='clean label')
    plt.plot(IS_NOISE,noisy2[:,0],'ro',markersize=5,label='noisy label')
    if args.mode not in ['symmetric','asymmetric','fairflip']:
        for i in range(1,n-1):
            plt.plot(NOT_NOISE,clean2[:,i],'bo',markersize=5)
            plt.plot(IS_NOISE,noisy2[:,i],'ro',markersize=5)
    else:
        plt.plot(NOT_NOISE,clean2[:,1],'bo',markersize=5)
        plt.plot(IS_NOISE,noisy2[:,1],'ro',markersize=5)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(DIR)
    # save_log={}
    # save_log['pi']=log2.tolist()
    # save_log['dpi']=log.tolist()
    # with open(DIR2,'w') as json_file:
    #     json.dump(save_log,json_file)


def plot_mu2(out,out2,true_noise,args):
    DIR1='./res/{}_{}/{}_test_mu3.png'.format(args.mode,args.ER,args.id)
    DIR2='./res/{}_{}/{}_confusion.png'.format(args.mode,args.ER,args.id)
    img1=out['D1']
    img2=out['D2']
    img3=out['D3']
    img4= out['confusion_matrix']
    img5=out2['D1']
    img6=out2['D2']
    img7=out2['D3']
    img8= out2['confusion_matrix']
    for i in range(10):
        img1[i,:]=img1[i,:]/(np.sum(img1[i,:]))
        img2[i,:]=img2[i,:]/(np.sum(img2[i,:]))
        img3[i,:]=img3[i,:]/(np.sum(img3[i,:]))
        img4[i,:]=img4[i,:]/(np.sum(img4[i,:]))
        img5[i,:]=img5[i,:]/(np.sum(img5[i,:]))
        img6[i,:]=img6[i,:]/(np.sum(img6[i,:]))
        img7[i,:]=img7[i,:]/(np.sum(img7[i,:]))
        img8[i,:]=img8[i,:]/(np.sum(img8[i,:]))
    img1=np.round(img1,2)
    img2=np.round(img2,2)
    img3=np.round(img3,2)
    img4=np.round(img4,2)
    img5=np.round(img5,2)
    img6=np.round(img6,2)
    img7=np.round(img7,2)
    img8=np.round(img8,2)
    img9=np.round(true_noise,2)
    img10=np.eye(10)
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    fig, ((ax3,ax4),(ax7,ax8)) = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('{} {} {}\% Transition Matrix \n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)
    # ax1.set_title('1st largest distribution clean',fontsize=25)
    # img = sns.heatmap(img1,ax=ax1,annot=True)
    # ax1.set_xlabel('predicted probablity',fontsize=20)
    # ax1.set_ylabel('predicted class',fontsize=20)
    
    # ax2.set_title('2nd largest distribution clean',fontsize=25)
    # img = sns.heatmap(img2,ax=ax2,annot=True)
    # ax2.set_xlabel('predicted probablity',fontsize=20)
    # ax2.set_ylabel('predicted class',fontsize=20)

    ax3.set_title('Clean - Predicted',fontsize=18)
    img = sns.heatmap(img3,ax=ax3,annot=True)
    ax3.set_xlabel('Clean Label',fontsize=15)
    ax3.set_ylabel('Noisy Label',fontsize=15)

    ax4.set_title('Clean - True',fontsize=18)
    img = sns.heatmap(img7,ax=ax4,annot=True)
    ax4.set_xlabel('Clean Label',fontsize=15)
    ax4.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()

    # ax5.set_title('1st largest distribution ambiguous',fontsize=25)
    # img = sns.heatmap(img5,ax=ax5,annot=True)
    # ax5.set_xlabel('predicted probablity',fontsize=20)
    # ax5.set_ylabel('predicted class',fontsize=20)
    
    # ax6.set_title('2nd largest distribution ambiguous',fontsize=25)
    # img = sns.heatmap(img6,ax=ax6,annot=True)
    # ax6.set_xlabel('predicted probablity',fontsize=20)
    # ax6.set_ylabel('predicted class',fontsize=20)

    ax7.set_title('Ambiguous - Predicted',fontsize=18)
    img = sns.heatmap(img7,ax=ax7,annot=True)
    ax7.set_xlabel('Clean Label',fontsize=15)
    ax7.set_ylabel('Noisy Label',fontsize=15)

    ax8.set_title('Ambiguous - True',fontsize=18)
    img = sns.heatmap(img9,ax=ax8,annot=True)
    ax8.set_xlabel('Clean Label',fontsize=15)
    ax8.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()
    fig.savefig(DIR1)

    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    # fig.suptitle('DirtyCIFAR10 {} {} confusion'.format(args.mode,args.ER),fontsize=20)
    # ax1.set_title('Confusion matrix clean')
    # img = sns.heatmap(img4,ax=ax1,annot=True)
    # ax1.set_ylabel('predicted class',fontsize=10)
    # ax1.set_xlabel('labels',fontsize=10)
    
    # ax2.set_title('True clean')
    # img = sns.heatmap(img10,ax=ax2,annot=True)

    # ax3.set_title('Confusion matrix ambiguous')
    # img = sns.heatmap(img8,ax=ax3,annot=True)
    # ax3.set_ylabel('predicted class',fontsize=10)
    # ax3.set_xlabel('labels',fontsize=10)
    
    # ax4.set_title('True ambiguous')
    # img = sns.heatmap(img9,ax=ax4,annot=True)
    # plt.tight_layout()
    # fig.savefig(DIR2)
    

def plot_sigma2(out,out2,args):
    DIR3='./res/{}_{}/{}_test_sigma.png'.format(args.mode,args.ER,args.id)

    sigma=out['sigma']
    sigma2 = out2['sigma']
    log = np.zeros(10)
    log2 = np.zeros(10)
    x=[i for i in range(10)]
    x_1=[i+0.2 for i in x]
    #x_2=[i+0.2 for i in x]
    if args.mode not in ['symmetric','rp','rs','fairflip','mixup']:
        IS_NOISE = [7,2,5,6,3]
        NOT_NOISE=[0,1,4,8,9]
    else:
        IS_NOISE=[i for i in range(10)]
        NOT_NOISE=[]
    for i in range(10):
        sig=sigma[str(i)]
        dist=np.asarray(sig).T
        try:
            mean_dist=np.mean(dist)
            log[i]=mean_dist
        except:
            log[i]=0
        sig2=sigma2[str(i)]
        dist=np.asarray(sig2).T
        try:
            mean_dist=np.mean(dist)
            log2[i]=mean_dist
        except:
            log2[i]=0
    x_2=[i-0.2 for i in NOT_NOISE]
    x_3 = [i-0.2 for i in IS_NOISE]
    clean2=np.take(log2,NOT_NOISE,axis=0)
    noisy2=np.take(log2,IS_NOISE,axis=0)
    MIN_SIG =np.min(log)
    MAX_SIG=np.max(log2)
    NAME=args.data
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    fig = plt.figure(figsize=(8, 5))
    plt.title('{} {} {}\% Aleatoric Uncertainty \n'.format(NAME,args.mode,NOISE_RATE),fontsize=10)
    plt.xticks(x)
    if args.mode=='asymmetric':
        plt.ylim((MIN_SIG-0.005,MAX_SIG+0.003))
    else:    
        plt.ylim((MIN_SIG-0.02,MAX_SIG+0.01))
    plt.xlabel("Class",fontsize=10)
    plt.ylabel("Alea Mean",fontsize=10)
    plt.bar(x,log, width=0.4, align='edge', label='clean instance',color='lightskyblue')
    plt.plot(x_1,log,'bo',markersize=5)
    #plt.plot(IS_NOISE,noisy,'ro',markersize=5)
    plt.bar(x,log2,width=-0.4, align='edge',label='ambiguous  instance',color='lightpink')
    plt.plot(x_2,clean2,'bo',markersize=5,label='clean label')
    plt.plot(x_3,noisy2,'ro',markersize=5,label='noisy label')
    plt.legend(bbox_to_anchor=(-0.1, 1),loc=1, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(DIR3)
    save_log_dict(args,log,log2)