cd ..
# python main.py --data cifar100 --ER 0.4 --mode asymmetric --lr 2e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 40 --id 2 --mixup --alpha 4 --gpu 5
# python main.py --data cifar100 --ER 0.2 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 75 --id 3 --mixup --alpha 4 --gpu 5

# python main.py --data cifar100 --ER 0.2 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 75 --id 4 --mixup --alpha 4 --gpu 5
# python main.py --data cifar100 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 75 --id 3 --mixup --alpha 4 --gpu 5
# python main.py --data cifar100 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 75 --id 4 --mixup --alpha 4 --gpu 5


# python main.py --data cifar100 --ER 0.2 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 5 --mixup --alpha 4 --gpu 5
# python main.py --data cifar100 --ER 0.2 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 50 --id 6 --mixup --alpha 4 --gpu 6
python main.py --data cifar10 --mode instance --ER 0.2 --lr 5e-4 --wd 1e-4 --ratio 5 --ratio2 1 --lr_rate 0.1 --lr_step 50 --k 20 --gpu 5 --id 15 --mixup
python main.py --data cifar10 --mode instance --ER 0.2 --lr 5e-4  --wd 1e-4 --ratio 10 --ratio2 1 --lr_rate 0.1 --lr_step 50 --k 20 --gpu 5 --id 16 --mixup
# python main.py --data cifar10 --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 3 --mixup --alpha 6 --gpu 7
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 0.1 --ratio2 1 --ER 0.5 --lr_rate 0.2 --k 20 --gpu 2 --id 2
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 0.1 --ratio2 1 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 2 --id 2
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 0.1 --ratio2 1 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --gpu 2 --id 2
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 0.1 --ratio2 1 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 2 --id 2

# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 10 --ratio2 1 --ER 0.5 --lr_rate 0.2 --k 20 --id 3 --gpu 2
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 10 --ratio2 1 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 2 --id 3
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 10 --ratio2 1 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --gpu 2 --id 3
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 10 --ratio2 1 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 2 --id 3


# python main.py --data mnist --mode instance --ER 0.2 --id 10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --lr_step 3 --epoch 20 --k 20 --gpu 7 --id 1
# python main.py --data mnist --mode instance --ER 0.4 --id 10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --lr_step 3 --epoch 20 --k 20 --gpu 7 --id 21 --sig_min 1 --sig_max 2
# # python main.py --data cifar10 --mode instance --ER 0.4 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20 --gpu 7 --id 1
# python main.py --data cifar10 --mode instance --ER 0.3 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20 --gpu 7 --id 1

# python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.5 --lr_rate 0.2 --k 20 --gpu 0
# python main.py --data dirty_cifar10 --resnet --lr_step 150 --id 2 --lr 2e-2 --wd 5e-4 --ratio 1 --ratio2 1 --ER 0.5 --lr_rate 0.1 --k 20 --gpu 0 --batch_size 64 --epoch 300

# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 0 --ratio 1 --id 111 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 0 --ratio 0.5 --id 112 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 0 --ratio 0 --id 41 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 0 --ratio 0 --id 51 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 0 --ratio 0 --id 41 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 0 --ratio 0 --id 11 --k 1 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 0 --ratio 0 --id 21 --k 1 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 0 --ratio 0 --id 11 --k 1 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500
