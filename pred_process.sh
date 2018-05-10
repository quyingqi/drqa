python predict.py -test data/sogou_shuffle_valid.json \
-device 3 \
-dict saved_checkpoint/bilstm_last_shuffle-2/bilstm_last_shuffle-2.best.query.pre.dict \
-model saved_checkpoint/bilstm_last_shuffle-2/bilstm_last_shuffle-2.best.query.pre.model \
-output output/result
