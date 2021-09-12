# python main.py --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1
# python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1
# python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1
# python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1

# python main.py --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 0.1 --lr_rate 0.2 --id 102 --k 20 --cross_validation 1
# python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 0.1 --lr_rate 0.2 --id 102 --k 20 --cross_validation 1
# python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 0.1 --lr_rate 0.2 --id 102 --k 20 --cross_validation 1
# python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 0.1 --lr_rate 0.2 --id 102 --k 20 --cross_validation 1


# python main.py --mode asymmetric2 --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1
# python main.py --mode asymmetric3 --ER 0.6 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --id 101 --k 20 --cross_validation 1

# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.5 --lr_rate 0.2 --id 3 --k 20
# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.2 --lr_rate 0.2 --id 3 --k 20
# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.4 --mode asymmetric --lr_rate 0.2 --id 3 --k 20
# python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.8 --lr_rate 0.2 --id 3 --k 20

# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 24 --id 2 --ER 0.2
# python main.py --data cifar100 --ratio 0.1 --ratio2 0.1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 24 --id 2 --ER 0.4 --mode asymmetric

python main.py --data dirty_mnist --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.5 --lr_rate 0.2 --id 2
python main.py --data dirty_mnist --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.2 --lr_rate 0.2 --id 2

# python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0 --ratio2 1 --lr_rate 0.2 --id 5
# python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 0 --ratio2 1 --lr_rate 0.2 --id 5

# python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 101 --ER 0.2 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 101 --ER 0.5 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 101 --ER 0.8 --cross_validation 1
# python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 101 --ER 0.4 --mode asymmetric --cross_validation 1

# python main.py --data cifar100 --ratio 0.1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 102 --ER 0.2 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 102 --ER 0.5 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 102 --ER 0.8 --cross_validation 1
# python main.py --data cifar100 --ratio 0.1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --id 102 --ER 0.4 --mode asymmetric --cross_validation 1