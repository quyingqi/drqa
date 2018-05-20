#python predict.py -test data/sogou_valid.json \
python predict_my.py -test data/sogou_shuffle_valid.json \
-valid-data data/sogou_shuffle_valid-5.pt \
-device 0 \
-model saved_checkpoint/ee_feature-2/ee_feature-2.best.char.f1.model \
-output output/ee_feature_old \
-question \
