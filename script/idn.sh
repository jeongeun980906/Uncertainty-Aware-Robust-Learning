cd ..
python main.py --data cifar10 --mode instance --ER 0.2 --lr 5e-4 --wd 1e-4 --lr_rate 0.1 --lr_step 100
python main.py --data cifar10 --mode instance --ER 0.4 --lr 1e-3 --wd 1e-4 --lr_rate 0.1 --lr_step 20 --ratio 10 

python main.py --data mnist --mode instance --ER 0.2 --epoch 20 --lr 1e-3 --wd 1e-4 --lr_rate 0.1 --lr_step 10
python main.py --data mnist --mode instance --ER 0.4 --epoch 20 --lr 1e-3 --wd 1e-4 --lr_rate 0.1 --lr_step 10
