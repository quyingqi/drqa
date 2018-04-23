python predict.py -test data/sogou_valid.json \
-device 7 \
-dict model/bilstm_last_aligned-0/bilstm_last_aligned-0.best.query.pre.dict \
-model model/bilstm_last_aligned-0/bilstm_last_aligned-0.best.query.pre.model \
-output output/result
