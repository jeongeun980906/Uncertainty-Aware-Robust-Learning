cd ..
python main.py --data cifar100 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 5 --mixup --alpha 4 --gpu 7
python main.py --data cifar100 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-7 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 50 --id 6 --mixup --alpha 4 --gpu 7

# python main.py --data cifar10 --ER 0.2 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --lr_step 100 --k 20 --gpu 3 --id 5 --mixup --alpha 4

# python main.py --data cifar10 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 75 --id 4 --mixup --alpha 4 --gpu 4
# python main.py --data cifar10 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 3 --mixup --alpha 6 --gpu 4
# python main.py --data cifar10 --mode instance --ER 0.8 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20 --gpu 3

# python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --gpu 3
# python main.py --data dirty_cifar10 --lr_step 150 --resnet --id 2 --lr 2e-2 --wd 5e-4 --ratio 1 --ratio2 1 --ER 0.4 --mode asymmetric --lr_rate 0.1 --k 20 --gpu 3 --batch_size 64 --epoch 300

# # python main.py --data cifar10 --ER 0.2 --lr 2e-2 --gpu 3 --ratio 0.1 --id 116 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --gpu 3 --ratio 0.1 --id 116 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 3 --ratio 0.1 --id 43 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 3 --ratio 0.1 --id 53 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 3 --ratio 0.1 --id 43 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 3 --ratio 0.1 --id 13 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 3 --ratio 0.1 --id 23 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 3 --ratio 0.1 --id 13 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 0 --ratio 0 --id 4 --k 1 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 0 --ratio 0 --id 14 --k 1 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 0 --ratio 0 --id 24 --k 1 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 0 --ratio 0 --id 34 --k 1 --lr_rate 0.1 --mixup --alpha 4

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0 --id 37 --k 2 --lr_rate 0.1 --mixup --alpha 4