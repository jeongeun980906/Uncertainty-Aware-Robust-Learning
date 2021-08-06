import math

def lambda_tunner(n_class):
    return {'mnist':[1,1],'trec':[5/n_class,1/n_class],'cifar10':[5/n_class,1/n_class],'cifar100':[5/n_class,1/n_class]}