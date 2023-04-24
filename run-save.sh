# python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class road --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA

# # DSTA
# python main.py --model DSTA --dataset ours --phase train --batch_size 64 --data_class road --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name DSTA

# # without soft label
# python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class road --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_softlabel

# # without gap
# python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class road --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 0 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_gap

# # without gapframe
# python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class road --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 0 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_gapframe

python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class freeway --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA

# DSTA
python main.py --model DSTA --dataset ours --phase train --batch_size 64 --data_class freeway --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name DSTA

# without soft label
python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class freeway --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_softlabel

# without gap
python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class freeway --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 0 --loss_gap_frame 10 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_gap

# without gapframe
python main.py --model IDSTA --dataset ours --phase train --batch_size 64 --data_class freeway --is_save --feature_name vgg16 --ptm_dataset cars --box_type od --epoch 30 --toa_modify 0 --loss_ce 0.05 --loss_gap 30 --loss_gap_frame 0 --loss_beta 1 --base_lr 0.001 --lamda 5 --aug_type_num 12 --output_dir /data/yehj/SAVES/DSTA/output/ --test_iter 51 --tpt 0.03 --optim Adam --schedule cosine --gpu 0 --save_name IDSTA_without_gapframe