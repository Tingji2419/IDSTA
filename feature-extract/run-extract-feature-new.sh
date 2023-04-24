# source activate py37
for seed in {0..11}
do
echo $seed
python extract_feature_new.py --model vgg16 --data_class freeway --dataloader_type train --box_type od --gpu 0 --ptm_dataset cars --seed $seed
python extract_feature_new.py --model vgg16 --data_class road --dataloader_type train --box_type od --gpu 0 --ptm_dataset cars --seed $seed
done

