#python predict.py -test data/sogou_valid.json \
python predict_my.py -test data/sogou_shuffle_valid.json \
-device 2 \
-model saved_checkpoint/celoss/celoss.best.char.f1.model \
-question \
-output output/result
