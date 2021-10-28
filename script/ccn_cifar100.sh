cd ..
python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --ER 0.2
python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --ER 0.5
python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --ER 0.8
python main.py --data cifar100 --ratio 1 --ratio2 1 --lr 1e-3 --wd 1e-7 --lr_rate 0.2 --k 20 --gpu 0 --ER 0.4 --mode asymmetric
