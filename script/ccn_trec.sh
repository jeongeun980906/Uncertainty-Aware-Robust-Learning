cd ..

python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.2 --sampler True --ratio 1 --ratio2 1 --epoch 100 --k 20
python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.5 --sampler True --ratio 1 --ratio2 1 --epoch 100 --k 20
python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.7 --sampler True --ratio 1 --ratio2 1 --epoch 100 --k 20
python main.py --data trec --lr 0.1 --wd 1e-4 --lr_rate 0.9 --ER 0.4 --mode asymmetric --sampler True --ratio 1 --ratio2 1 --epoch 100 --k 20
