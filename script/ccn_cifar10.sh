cd ..

python main.py --data cifar10 --ER 0.2 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --data cifar10 --ER 0.5 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --data cifar10 --ER 0.8 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --data cifar10 --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --data cifar10 --mode asymmetric2 --ER 0.4 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --data cifar10 --mode asymmetric3 --ER 0.6 --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
