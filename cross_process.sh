python train.py -epoch 10 -brnn -batch 100 -lr 0.005 \
	-device 0 \
	-exp-name cross_4\
	-train-data data/cross_train-4.pt \
	-valid-data data/cross_valid-4.pt

python train.py -epoch 5 -brnn -batch 100 -lr 0.0005 \
	-device 0 \
	-exp-name cross_4_finetune \
	-train-data data/cross_train-4.pt \
	-valid-data data/cross_valid-4.pt \
	-resume_snapshot saved_checkpoint/cross_4/cross_4.best.query.pre.model

cd answer_select
python train.py -epoch 10 -brnn -batch 40 -lr 0.001 \
	-device 0 \
	-exp-name cross_4_finetune_ranking \
	-baidu-data ../data/baidu_data.pt \
	-train-data ../data/cross_train-4.pt \
	-valid-data ../data/cross_valid-4.pt \
	-resume_snapshot ../saved_checkpoint/cross_4_finetune/cross_4_finetune.best.query.pre.model

