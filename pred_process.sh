#python predict.py -test data/sogou_valid.json \
python predict_my.py -test data/sogou_shuffle_valid.json \
-device 3 \
-model saved_checkpoint/remove_question_att/remove_question_att.best.query.pre.model \
-output output/tmp
-question \
