python train.py \
-word-vectors ../data/penny.cbow.dim300.bin \
-dict ../data/vocab.pt \
-epoch 10 \
-batch 40 \
-device 1 \
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
-exp-name cross_4_ranking \
-baidu-data ../data/baidu_data.pt \
-train-data ../data/cross_train-4.pt \
-valid-data ../data/cross_valid-4.pt \
-resume_snapshot ../saved_checkpoint/cross_4/cross_4.best.query.pre.model
