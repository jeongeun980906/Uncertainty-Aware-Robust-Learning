cd ..

python main.py --ER 0.2 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --ER 0.5 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --ER 0.8 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --mode asymmetric --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --mode asymmetric2 --ER 0.4 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
python main.py --mode asymmetric3 --ER 0.6 --lr 1e-3 --wd 1e-4 --epoch 20 --ratio 1 --ratio2 1 --lr_rate 0.2 --k 20
