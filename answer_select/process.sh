python train.py \
-word-vectors ../data/penny.cbow.dim300.bin \
-dict ../data/vocab.pt \
-epoch 10 \
-batch 25 \
-device 2 \
-pos-vec-size 5 \
-ner-vec-size 5 \
-hidden-size 128 \
-optimizer Adamax \
-lr 0.005 \
-num-layers 3 \
-loss_margin 1 \
-brnn \
-rnn-type LSTM \
-multi-layer last \
-exp-name load_ranking_s_e_2 \
-baidu-data ../drqa-classify/data/baidu_data.pt \
-train-data ../drqa-classify/data/sogou_shuffle_train.pt \
-valid-data ../drqa-classify/data/sogou_shuffle_valid.pt \
-resume_snapshot ../saved_checkpoint/feature_c-finetune/feature_c-finetune.best.query.pre.model
