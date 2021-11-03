from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.stats import kendalltau

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

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
    plt.show()

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

def save_log_dict(args,train_acc,test_acc):
    DIR_PATH='./res/{}_{}_{}/{}_log.json'.format(args.data,args.mode,args.ER,args.id)
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

def plot_tm_ccn(out,args,true_noise,labels=10):
    DIR1='./res/{}_{}_{}/{}_test_tm.png'.format(args.data,args.mode,args.ER,args.id)
    DIR2='./res/{}_{}_{}/{}_test.json'.format(args.data,args.mode,args.ER,args.id)
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
            raise
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
        fig.suptitle('{} {} {}% Transition Matrix'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=30)
        ax3.set_title('Predicted',fontsize=25)
        img = sns.heatmap(img3,ax=ax3, cmap="YlGnBu", vmin=0, vmax=1)
        ax3.set_xlabel('Clean Label',fontsize=20)
        ax3.set_ylabel('Noisy Label',fontsize=20)

        ax4.set_title('True',fontsize=25)
        img = sns.heatmap(img7,ax=ax4, cmap="YlGnBu", vmin=0, vmax=1)
        ax4.set_xlabel('Clean Label',fontsize=20)
        ax4.set_ylabel('Noisy Label',fontsize=20)
        plt.tight_layout()
        fig.savefig(DIR1)

    else:
        #fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 16))
        fig, (ax3,ax4) = plt.subplots(2, 1, figsize=(6, 12))
        fig.suptitle('{} {} {}% Transition Matrix'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)

        ax3.set_title('Predicted',fontsize=18)
        img = sns.heatmap(img3,ax=ax3,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        ax3.set_xlabel('Clean Label',fontsize=15)
        ax3.set_ylabel('Noise Label',fontsize=15)

        ax4.set_title('True',fontsize=18)
        img = sns.heatmap(img7,ax=ax4,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        ax4.set_xlabel('Clean Label',fontsize=15)
        ax4.set_ylabel('Noisy Label',fontsize=15)
        plt.tight_layout()
        fig.savefig(DIR1)
    save_log = {}
    save_log['TM']=img3.tolist()
    save_log['GT']=img7.tolist()
    with open(DIR2,'w') as json_file:
        json.dump(save_log,json_file,indent=4)

def plot_hist(clean_eval,ambiguous_eval,args):
    roc ={}
    plt.figure(figsize=(10,10))
    NAME=args.data
    NAME=NAME[6:].upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    size = len(clean_eval['alea_'])
    GT=[0 if i<size else 1 for i in range(2*size)]
    UTYPE = ['alea_','epis_','pi_entropy_','maxsoftmax_','entropy_']
    title = ['Aleatoric','Epistemic','$\pi$ Entropy','MaxSoftmax', 'Softmax Entropy']
    plt.suptitle('Dirty {} {} {}% Uncertainty Histogram\n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=10)
    for i,u in enumerate(UTYPE):
        plt.subplot(3,2,i+1)
        plt.title(title[i])
        plt.hist(clean_eval[u], color='b',label='clean',bins=100, alpha=0.5)
        plt.hist(ambiguous_eval[u],color='r',label='ambiguous',bins=100, alpha=0.5)
        plt.legend()
        EVAL=clean_eval[u]+ambiguous_eval[u]
        roc[u]=roc_auc_score(GT,EVAL)
    plt.savefig('./res/{}_{}_{}/{}_hist.png'.format(args.data,args.mode,args.ER,args.id))
    return roc

def plot_tm_sdn(out,out2,true_noise,args):
    DIR1='./res/{}_{}_{}/{}_test_tm.png'.format(args.data,args.mode,args.ER,args.id)
    DIR2='./res/{}_{}_{}/{}_test.json'.format(args.data,args.mode,args.ER,args.id)
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
    NAME=args.data[6:]
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    NOISE_TYPE= args.mode
    NOISE_TYPE=NOISE_TYPE.capitalize()
    fig, ((ax3,ax7),(ax4,ax8)) = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Dirty {} {} {}% Transition Matrix \n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)
    ax3.set_title('Clean - Predicted',fontsize=18)
    img = sns.heatmap(img3,ax=ax3,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax3.set_xlabel('Clean Label',fontsize=15)
    ax3.set_ylabel('Noisy Label',fontsize=15)

    ax4.set_title('Clean - True',fontsize=18)
    img = sns.heatmap(img10,ax=ax4,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax4.set_xlabel('Clean Label',fontsize=15)
    ax4.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()

    ax7.set_title('Ambiguous - Predicted',fontsize=18)
    img = sns.heatmap(img7,ax=ax7,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax7.set_xlabel('Clean Label',fontsize=15)
    ax7.set_ylabel('Noisy Label',fontsize=15)

    ax8.set_title('Ambiguous - True',fontsize=18)
    img = sns.heatmap(img9,ax=ax8,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax8.set_xlabel('Clean Label',fontsize=15)
    ax8.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()
    fig.savefig(DIR1)
    
    save_log={}
    save_log['TM_clean']=img3.tolist()
    save_log['GT_clean']=img10.tolist()
    save_log['TM_amb']=img7.tolist()
    save_log['GT_amb']=img9.tolist()
    with open(DIR2,'w') as json_file:
        json.dump(save_log,json_file,indent=4)

def plot_alea_sdn(out,out2,args):
    DIR3='./res/{}_{}_{}/{}_test_alea.png'.format(args.data,args.mode,args.ER,args.id)
    DIR2='./res/{}_{}_{}/{}_test.json'.format(args.data,args.mode,args.ER,args.id)
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
    NAME=args.data[6:]
    NAME=NAME.upper()
    NOISE_RATE=int(args.ER*100)
    fig = plt.figure(figsize=(8, 5))
    plt.title('{} {} {}% Aleatoric Uncertainty \n'.format(NAME,args.mode,NOISE_RATE),fontsize=10)
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
    with open(DIR2,'r') as json_file:
        save_log = json.load(json_file)
    save_log['alea_clean']=log.tolist()
    save_log['alea_amb']=log2.tolist()
    with open(DIR2,'w') as json_file:
        json.dump(save_log,json_file,indent=4)