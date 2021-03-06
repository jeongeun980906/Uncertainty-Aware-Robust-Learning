cd ..
python main.py --data cifar100 --ER 0.8 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 2 --sig_min 1 --k 20 --lr_step 50 --id 5 --mixup --alpha 4 --gpu 6
python main.py --data cifar100 --ER 0.8 --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --lr_rate 0.1 --sig_max 1 --sig_min 0.1 --k 20 --lr_step 50 --id 6 --mixup --alpha 4 --gpu 6

# python main.py --data cifar10 --ER 0.8 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 8 --mixup --alpha 1 --gpu 3
# python main.py --data cifar10 --ER 0.8 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20

# python main.py --data cifar10 --ER 0.8 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 3 --mixup --alpha 6 --gpu 5
# python main.py --data cifar10 --mode instance --ER 0.4 --lr 1e-3 --wd 1e-4 --ratio 5 --id 2 --ratio2 1 --lr_rate 0.2 --k 20 --gpu 2

# python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 2
# python main.py --data dirty_cifar10 --lr_step 150 --epoch 300 --resnet --id 2 --lr 2e-2 --wd 5e-4 --ratio 1 --ratio2 1 --ER 0.8 --lr_rate 0.1 --k 20 --gpu 2 --batch_size 64

# python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.8 --lr_rate 0.2 --k 20 --gpu 2
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 2 --ratio 0.1 --id 113 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 2 --ratio 0.1 --id 113 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 2 --ratio 0 --id 52 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 2 --ratio 0 --id 62 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 2 --ratio 0 --id 52 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 2 --ratio 0 --id 22 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 2 --ratio 0 --id 32 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 2 --ratio 0 --id 22 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 2 --ratio 0.5 --id 3 --k 5 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 2 --ratio 0.5 --id 13 --k 5 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 2 --ratio 0.5 --id 23 --k 5 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 2 --ratio 0.5 --id 33 --k 5 --lr_rate 0.1 --mixup --alpha 4

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0 --id 27 --k 2 --lr_rate 0.1 --mixup --alpha 0.5
