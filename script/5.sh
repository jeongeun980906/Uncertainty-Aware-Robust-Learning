cd ..
python main.py --data cifar10 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 31 --mixup --alpha 4 --gpu 3
python main.py --data cifar10 --ER 0.4 --mode asymmetric --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 50 --id 32 --mixup --alpha 4 --gpu 3

python main.py --data cifar10 --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --sig_max 2 --sig_min 1 --lr_rate 0.1 --k 20 --lr_step 50 --id 31 --mixup --alpha 4 --gpu 3
python main.py --data cifar10 --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --sig_max 1 --sig_min 0.1 --lr_rate 0.1 --k 20 --lr_step 50 --id 32 --mixup --alpha 4 --gpu 3

# python main.py --data cifar10 --mode instance --ER 0.2 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --lr_step 100 --k 20 --gpu 3 --id 5 --mixup --alpha 4
# python main.py --data cifar10 --mode instance --ER 0.4 --lr 5e-4 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.1 --lr_step 100 --k 20 --gpu 3 --id 6 --mixup --alpha 4 --gpu 6

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --gpu 4 --ratio 1 --id 114 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --gpu 4 --ratio 0.5 --id 115 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 4 --ratio 0.1 --id 63 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 4 --ratio 0.1 --id 63 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 4 --ratio 0 --id 51 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 4 --ratio 0.1 --id 33 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 4 --ratio 0.1 --id 33 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 4 --ratio 0 --id 21 --k 1 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0.1 --id 5 --k 2 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0.1 --id 15 --k 2 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0.1 --id 25 --k 2 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0.1 --id 35 --k 2 --lr_rate 0.1 --mixup --alpha 4

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0 --id 7 --k 2 --lr_rate 0.1