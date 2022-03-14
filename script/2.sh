cd ..
python main.py --data dirty_cifar10 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.2 --k 20 --gpu 1
# python main.py --data dirty_cifar10 --id 2 --lr 2e-2 --wd 5e-4 --lr_step 100 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.1 --k 20 --gpu 1 --batch_size 64 --resnet
python main.py --data dirty_cifar10 --id 2 --lr 2e-2 --wd 5e-4 --lr_step 150 --ratio 1 --ratio2 1 --ER 0.2 --lr_rate 0.1 --k 20 --gpu 1 --batch_size 64 --resnet --epoch 300

# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 61 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256
# python main.py --data cifar10 --ER 0.2 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 42 --k 5 --lr_rate 0.5 --mixup --alpha 4 --epoch 500
# python main.py --data cifar10 --ER 0.5 --lr 5e-3 --sigma --gpu 1 --ratio 0 --id 61 --k 1 --lr_rate 0.5 --mixup --alpha 4 --epoch 500 --batch_size 256

# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 1 --ratio 1 --id 111 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64
# python main.py --data cifar10 --ER 0.2 --lr 2e-2 --sigma --gpu 1 --ratio 0.5 --id 112 --k 20 --lr_rate 0.1 --lr_step 150 --mixup --alpha 4 --epoch 500 --batch_size 64

# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 2 --k 5 --lr_rate 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 12 --k 5 --lr_rate 0.1 --mixup --alpha 0.1
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 22 --k 5 --lr_rate 0.1 --mixup --alpha 0.5
# python main.py --data cifar10 --ER 0.5 --lr 1e-2 --sigma --gpu 1 --ratio 0.1 --id 32 --k 5 --lr_rate 0.1  --mixup --alpha 4