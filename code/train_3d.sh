python train_3d.py --dataset_name LA --model DCPA3d --labelnum 16 --gpu 0 && \
python train_3d.py --dataset_name LA --model DCPA3d --labelnum 8 --gpu 0 && \
python train_3d.py --dataset_name LA --model DCPA3d --labelnum 4 --gpu 0 && \

python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labelnum 16 --gpu 0 && \
python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labelnum 8 --gpu 0 && \
python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labelnum 4 --gpu 0 && \

python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labelnum 12 --gpu 0 && \
python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labelnum 6 --gpu 0 && \
python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labelnum 3 --gpu 0 && \

python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labelnum 12 --gpu 0 && \
python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labelnum 6 --gpu 0 && \
python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labelnum 3 --gpu 0
