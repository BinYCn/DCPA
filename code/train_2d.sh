python train_2d.py  --model DCPA2d --labelnum 14 --gpu 0 && \
python train_2d.py  --model DCPA2d --labelnum 7 --gpu 0 && \
python train_2d.py  --model DCPA2d --labelnum 3 --gpu 0 && \

python test_2d.py --exp DCPA2d --model DCPA2d --labelnum 14 --gpu 0 && \
python test_2d.py --exp DCPA2d --model DCPA2d --labelnum 7 --gpu 0 && \
python test_2d.py --exp DCPA2d --model DCPA2d --labelnum 3 --gpu 0
