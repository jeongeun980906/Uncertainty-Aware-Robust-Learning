cd ..
# python main.py --data cifar100 --ER 0.8 --lr 2e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 2 --sig_min 1 --k 20 --lr_step 40 --id 2 --mixup --alpha 4 --gpu 4
# python main.py --data cifar100 --ER 0.5 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 2 --sig_min 1 --k 20 --lr_step 75 --id 3 --mixup --alpha 4 --gpu 4

# python main.py --data cifar100 --ER 0.5 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 1 --sig_min 0.1 --k 20 --lr_step 75 --id 4 --mixup --alpha 4 --gpu 4
# python main.py --data cifar100 --ER 0.8 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 2 --sig_min 1 --k 20 --lr_step 75 --id 3 --mixup --alpha 4 --gpu 4
# python main.py --data cifar100 --ER 0.8 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 1 --sig_min 0.1 --k 20 --lr_step 75 --id 4 --mixup --alpha 4 --gpu 4

# python main.py --data cifar100 --ER 0.5 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 2 --sig_min 1 --k 20 --lr_step 50 --id 5 --mixup --alpha 4 --gpu 4
# python main.py --data cifar100 --ER 0.5 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 1 --sig_min 0.1 --k 20 --lr_step 50 --id 6 --mixup --alpha 4 --gpu 4


# python main.py --data cifar10 --ER 0.4 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20 --lr_step 20 --id 5 --mixup --alpha 4 --gpu 3

python main.py --data cifar10 --mode instance --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 5 --ratio2 1 --lr_rate 0.1 --lr_step 20 --k 20 --gpu 7 --id 13
python main.py --data cifar10 --mode instance --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 10 --ratio2 1 --lr_rate 0.1 --lr_step 20 --k 20 --gpu 7 --id 14
# python main.py --data cifar10 --ER 0.5 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 100 --id 5 --mixup --alpha 4 --gpu 6

# python main.py --data cifar10 --ER 0.5 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 3 --mixup --alpha 6 --gpu 6
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 0.1 --ER 0.5 --lr_rate 0.2 --k 20 --id 4 --gpu 3
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 0.1 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 3 --id 4
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 0.1 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --gpu 3 --id 4
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 0.1 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 3 --id 4

# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 10 --ER 0.5 --lr_rate 0.2 --k 20 --id 5 --gpu 3
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 10 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 3 --id 5
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 10 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --gpu 3 --id 5
# python main.py --data dirty_cifar10 --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 10 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 3 --id 5
# python main.py --ER 0.2 --lr 1e-3 --mode instance --wd 1e-4 --epoch 20 --ratio 5 --id 2 --ratio2 1 --lr_rate 0.5 --k 20 --lr_step 3 --gpu 1
# python main.py --ER 0.5 --lr 1e-3 --mode instance --wd 1e-4 --epoch 20 --ratio 5 --id 2 --ratio2 1 --lr_rate 0.5 --k 20 --lr_step 3 --gpu 1
# python main.py --ER 0.8 --lr 1e-3 --mode instance --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.5 --k 20 --lr_step 3 --gpu 1
# python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 1
# python main.py --data dirty_cifar10 --id 2 --lr 2e-2 --wd 5e-4 --lr_step 100 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.1 --k 20 --gpu 1 --batch_size 64 --resnet
# python main.py --data dirty_cifar10 --id 2 --lr 2e-2 --wd 5e-4 --lr_step 150 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.1 --k 20 --gpu 1 --batch_size 64 --resnet --epoch 300

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 61 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 42 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 61 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 1 --ratio 1 --id 111 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 1 --ratio 0.5 --id 112 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 2 --k 5 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 12 --k 5 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 22 --k 5 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 32 --k 5 --lr_rate 0.1  --mixup --alpha 4