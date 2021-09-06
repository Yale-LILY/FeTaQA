# T5 Details
## Directory Structure
```
├── Args.py # command line arguments/config definitions
├── configs # model training and prediction config
│   ├── t5-large.json
│   └── t5-small.json
├── data # preprocessed dataset
├── dataset_format.py # formats dataset into json formats in data/ folder
├── env.yml
├── outputs # t5 model predictions on test set
│   ├── t5-base-test-predictions.txt
│   ├── t5-large-test-predictions.txt
│   └── t5-small-test-predictions.txt
├── README.md
├── train.py # main script for train/dev/test
└── utils.py # table linearization strategies

```

## Naive Model Parallelism
Uncomment the lines at https://github.com/Yale-LILY/FetaQA-models/blob/7bbd9c994f38fbec085fc3180e9d2720d2d1e10d/end2end/train.py#L145 to enable


## Relevant configs:
1. __training configs__:
  - "output_dir":[checkpoints directory \<rootdir>/checkpoints/<experiment_id>],
  - "overwrite_output_dir":false,
  - "do_train" : true, 
    set to true to train
  - "do_eval":true, __set to true to evaluate__
  - "do_predict":false,
  - "num_train_epochs":13,            __modify epochs (caculation: epochs*(dataset_size/batch_size) = train steps)__
  - "per_device_train_batch_size":8,  __modified this to fit memory__
  - "per_device_eval_batch_size":64,   
  - "warmup_steps":500,                
  - "weight_decay":0.01,               
  - "predict_with_generate" : true   __Whether to use generate to calculate generative metrics (ROUGE, BLEU).__
  - Other potentially relevant arguments
    - [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
    - [--adam_epsilon ADAM_EPSILON]
    - [--max_grad_norm MAX_GRAD_NORM]
    - [--max_steps MAX_STEPS]
    - [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
    - [--save_steps SAVE_STEPS]


2. __model config__:
  - "model_name_or_path":"t5-small"
    <br>__try other model names from huggingface__

3. __data config__:
  - "train_file":[path to train set json],
  - "validation_file":[path to dev set json],
  - "test_file":[path to test set json]
  - "summary_column" : "answer",
      <br>__tells the data loader which json key to look at for tgt__
  - "text_column" : "table_array",
      <br>__tells the data loader which json key to look at for src__
  - "context_column" : "question",
     <br>__tells the data loader which json key to look at for context (appended to src in preprocessing step)__
  - "source_prefix" : "summarize: ",
     <br>__tells the data loader which json key to look at for context (appended to src in preprocessing step)__
  - "max_source_length" : 512,
      <br> __t5 max token?__
  - "max_target_length" : 25,
     <br> __adjust this to get suitable answer length__
  - "pad_to_max_length" : true
      <br>__setting to true uses default collator__




Check `python train.py -h` for more available arguments
