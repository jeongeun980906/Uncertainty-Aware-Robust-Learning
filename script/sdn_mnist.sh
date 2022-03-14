cd ..

python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.5 --lr_rate 0.2 --k 20 --lr_step 5 --gpu 5
python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.2 --lr_rate 0.2 --k 20 --lr_step 5 --gpu 5
python main.py --data dirty_mnist --lr 2e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.4 --mode asymmetric --lr_rate 0.2 --k 20 --lr_step 5 --gpu 5
python main.py --data dirty_mnist --lr 1e-3 --wd 1e-4 --ratio 1 --ratio2 1 --epoch 20 --ER 0.8 --lr_rate 0.2 --k 20 --lr_step 5 --gpu 5