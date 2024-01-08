python train_3d.py --dataset_name LA --model DCPA3d --labeled_num 16 --gpu 0 && \
python train_3d.py --dataset_name LA --model DCPA3d --labeled_num 8 --gpu 0 && \
python train_3d.py --dataset_name LA --model DCPA3d --labeled_num 4 --gpu 0 && \

python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labeled_num 16 --gpu 0 && \
python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labeled_num 8 --gpu 0 && \
python test_3d.py --dataset_name LA --model DCPA3d --exp DCPA3d --labeled_num 4 --gpu 0 && \

python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labeled_num 12 --gpu 0 && \
python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labeled_num 6 --gpu 0 && \
python train_3d.py --dataset_name Pancreas_CT --model DCPA3d --labeled_num 3 --gpu 0 && \

python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labeled_num 12 --gpu 0 && \
python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labeled_num 6 --gpu 0 && \
python test_3d.py --dataset_name Pancreas_CT --model DCPA3d --exp DCPA3d --labeled_num 3 --gpu 0
