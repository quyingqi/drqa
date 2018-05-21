python train.py \
-word-vectors data/penny.cbow.dim300.bin \
-epoch 10 \
-batch 32 \
-device 3 \
-pos-vec-size 5 \
-ner-vec-size 5 \
-hidden-size 128 \
-optimizer Adamax \
-lr 0.02 \
-num-layers 3 \
-brnn \
-rnn-type LSTM \
-multi-layer last \
-exp-name ee_feature-multi_answer \
-baidu-data data/baidu_data-7.pt \
-train-data data/sogou_shuffle_train-7.pt \
-valid-data data/sogou_shuffle_valid-7.pt
-resume_snapshot saved_checkpoint/celoss-score_sqrt/celoss-score_sqrt.best.loss.model \
