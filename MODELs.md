
## T5 end2end model
Script adapted from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_seq2seq.py

```
cd end2end
conda create env -f env.yml
conda activate fetaqa-e2e
```
Then, convert dataset format from jsonl to json `python dataset_format.py inputdir outputdir`. 

(Preprocessed version can be found in `end2end/data`)

Choose a config json file from `end2end/config`, then

    ```
    #supports multi-gpu
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python train.py configs/t5-large.json
    ```
## TAPAS Pipeline Model
To be released...
