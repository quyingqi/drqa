#python predict.py -test data/sogou_valid.json \
python predict_my.py -test data/sogou_shuffle_valid.json \
-valid-data data/sogou_shuffle_valid-9.pt \
-device 3 \
-model saved_checkpoint/feature_c-finetune/feature_c-finetune.best.char.f1.model answer_select/saved_checkpoint/load_ranking_s_e/load_ranking_s_e.best.char.f1.model drqa-classify/saved_checkpoint/load_ranking_s_e_fix/load_ranking_s_e_fix.best.char.f1.model \
-output output/ee_feature_old \
-question \
