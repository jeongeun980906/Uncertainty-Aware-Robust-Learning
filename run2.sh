# python main.py --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 1 --lr_rate 0.2 --id 103 --k 20 --cross_validation 1 --gpu 1
# python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 1 --lr_rate 0.2 --id 103 --k 20 --cross_validation 1 --gpu 1
# python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 1 --lr_rate 0.2 --id 103 --k 20 --cross_validation 1 --gpu 1
# python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 1 --lr_rate 0.2 --id 103 --k 20 --cross_validation 1 --gpu 1

# python main.py --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 0.1 --lr_rate 0.2 --id 104 --k 20 --cross_validation 1 --gpu 1
# python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 0.1 --lr_rate 0.2 --id 104 --k 20 --cross_validation 1 --gpu 1 
# python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 0.1 --lr_rate 0.2 --id 104 --k 20 --cross_validation 1 --gpu 1
# python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0.1 --ratio2 0.1 --lr_rate 0.2 --id 104 --k 20 --cross_validation 1 --gpu 1


# python main.py --data cifar10 --lr 1e-3 --wd 5e-4 --k 24 --tunner 1 --lr_rate 0.1
# python main.py --data cifar10 --lr 1e-3 --wd 5e-4 --k 24 --ratio 1 --ratio2 1 --lr_rate 0.1 --ER 0.5
# python main.py --data cifar10 --lr 1e-3 --wd 5e-4 --k 24 --ratio 1 --ratio2 1 --lr_rate 0.1 --ER 0.8
# python main.py --data cifar10 --lr 1e-3 --wd 5e-4 --k 24 --ratio 1 --ratio2 1 --lr_rate 0.1 --ER 0.4 --mode asymmetric

# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.4 --mode asymmetric --lr_rate 0.4
# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.4 --mode asymmetric2 --lr_rate 0.4

# python main.py --ER 0.6 --lr 1e-3 --wd 1e-4 --epoch 20 --gpu 0 --ratio 1 --ratio2 1 --lr_rate 0.4 --id 3
# python main.py --ER 0.7 --lr 1e-3 --wd 1e-4 --epoch 20 --gpu 0 --ratio 1 --ratio2 1 --lr_rate 0.4 --id 3
#python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --gpu 1 --ratio 1 --ratio2 1 --lr_rate 0.3 --id 3

# python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --gpu 1 --ratio 0 --ratio2 1 --lr_rate 0.4 --id 5
# python main.py --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-4 --epoch 20 --gpu 1 --ratio 0 --ratio2 1 --lr_rate 0.4 --id 5


# python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.2 --sampler True --ratio 1 --ratio2 1 --id 6 --epoch 100 --k 20 --gpu 1
# python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.5 --sampler True --ratio 1 --ratio2 1 --id 6 --epoch 100 --k 20 --gpu 1
# python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.7 --sampler True --ratio 1 --ratio2 1 --id 6 --epoch 100 --k 20 --gpu 1
# python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.4 --mode asymmetric --sampler True --ratio 1 --ratio2 1 --id 6 --epoch 100 --k 20 --gpu 1

# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 3 --ER 0.5
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 3 --ER 0.8

# python main.py --ER 0.6 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 6 --k 20 --gpu 1
# python main.py --ER 0.7 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 6 --k 20 --gpu 1

python main.py --data dirty_mnist --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.8 --lr_rate 0.2 --id 2
python main.py --data dirty_mnist --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.4 --mode asymmetric --lr_rate 0.2 --id 2

# python main.py --data cifar100 --ratio 1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 103 --ER 0.2 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 103 --ER 0.5 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 103 --ER 0.8 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 103 --ER 0.4 --mode asymmetric --gpu 1 --cross_validation 1
 
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 104 --ER 0.2 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 104 --ER 0.5 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 104 --ER 0.8 --gpu 1 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 104 --ER 0.4 --mode asymmetric --gpu 1 --cross_validation 1