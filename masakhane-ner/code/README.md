# african-ner

#### Example to run the NER code (i.e fine-tuning mBERT and XLM-R end-to-end)
#### Using Multi-Lingual BERT

```
export MAX_LENGTH=164
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=swa_bert
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=10000
export SEED=1
CUDA_VISIBLE_DEVICES=1,2,3 python3 train_ner.py --data_dir data/swa/ \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

#### Using XLM-RoBERTa
```
export MAX_LENGTH=164
export BERT_MODEL=xlm-roberta-base
export OUTPUT_DIR=swa_xlmr
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=10000
export SEED=1

CUDA_VISIBLE_DEVICES=1,2,3 python3 train_ner.py --data_dir data/swa/ \
--model_type xlmroberta \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

#### For the MeanE-BiLSTM model (i.e extract sentence embeddings from mBERT and XLM-R by taking the mean of the 12 LM layers before passing them into the BiLSTM + linear classifier)
#### Using Multi-Lingual BERT

```
export MAX_LENGTH=164
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=yor_freezembert
export BATCH_SIZE=32
export NUM_EPOCHS=50
export SAVE_STEPS=10000
export SEED=1

CUDA_VISIBLE_DEVICES=3 python3 train_ner_freezedBERT.py --data_dir data/yor/ \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS --learning_rate 5e-4 \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```


#### Using XLM-RoBERTa

```
export MAX_LENGTH=164
export BERT_MODEL=xlm-roberta-base
export OUTPUT_DIR=yor_freezexlmr
export BATCH_SIZE=32
export NUM_EPOCHS=50
export SAVE_STEPS=10000
export SEED=1

CUDA_VISIBLE_DEVICES=3 python3 train_ner_freezedBERT.py --data_dir data/yor/ \
--model_type xlmroberta \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS --learning_rate 5e-4 \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```



