cd ..
python main.py --data cifar10 --ER 0.5 --lr 2e-2 --gpu 5 --ratio 1 --id 114 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
python main.py --data cifar10 --ER 0.5 --lr 2e-2 --gpu 5 --ratio 0.5 --id 115 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 5 --ratio 0 --id 62 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 5 --ratio 0.1 --id 53 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 5 --ratio 0 --id 42 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500

# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 5 --ratio 0 --id 32 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 5 --ratio 0.1 --id 23 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.5 --lr 2e-2 --sigma --gpu 5 --ratio 0 --id 12 --k 5 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 6 --ratio 0.5 --id 6 --k 2 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 6 --ratio 0.5 --id 16 --k 2 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 6 --ratio 0.5 --id 26 --k 2 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 6 --ratio 0.5 --id 36 --k 2 --lr_rate 0.1 --mixup --alpha 4

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 5 --ratio 0 --id 17 --k 2 --lr_rate 0.1 --mixup --alpha 0.1