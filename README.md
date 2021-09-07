# FeTaQA: Free-form Table Question Answering

FeTaQA is a **F**r**e**e-form **Ta**ble **Q**uestion **A**nswering dataset with 10K Wikipedia-based {*table, question, free-form answer, supporting table cells*} pairs. It yields a more challenging table QA setting because it requires generating free-form text answers after retrieval, inference, and integration of multiple discontinuous facts from a structured knowledge source. Unlike datasets of generative QA over text in which answers are prevalent with copies of short text spans from the source, answers in our dataset are human-generated explanations involving entities and their high-level relations.

You can find more details, analyses, and baseline results in [our paper](https://arxiv.org/abs/2104.00369).

# Baselines

## T5 end2end model
Script adapted from [huggingface examples](https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_seq2seq.py).

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
More details about the config setup can be found [here](https://github.com/Yale-LILY/FeTaQA/tree/main/end2end).

## TAPAS Pipeline Model
To be released...


## Citation
```
@article{nan2021feta,
  title={FeTaQA: Free-form Table Question Answering},
  author={Linyong Nan and Chiachun Hsieh and Ziming Mao and Xi Victoria Lin and Neha Verma and Rui Zhang and Wojciech Kryściński and Nick Schoelkopf and Riley Kong and Xiangru Tang and Murori Mutuma and Ben Rosand and Isabel Trindade and Renusree Bandaru and Jacob Cunningham and Caiming Xiong and Dragomir Radev},
  journal={arXiv preprint arXiv:2104.00369},
  year={2021}
```
