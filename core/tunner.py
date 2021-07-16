import math

def lambda_tunner(n_class):
    return {'mnist':[10/n_class,10/n_class],'trec':[20/n_class,10/n_class],'cifar10':[30/n_class,10/n_class],'cifar100':[30/n_class,10/n_class]}