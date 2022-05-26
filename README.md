# FeTaQA: Free-form Table Question Answering

FeTaQA is a **F**r**e**e-form **Ta**ble **Q**uestion **A**nswering dataset with 10K Wikipedia-based {*table, question, free-form answer, supporting table cells*} pairs. It yields a more challenging table QA setting because it requires generating free-form text answers after retrieval, inference, and integration of multiple discontinuous facts from a structured knowledge source. Unlike datasets of generative QA over text in which answers are prevalent with copies of short text spans from the source, answers in our dataset are human-generated explanations involving entities and their high-level relations.

You can find more details, analyses, and baseline results in [our paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00446/109273/FeTaQA-Free-form-Table-Question-Answering).

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


## License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

The FeTaQA dataset is distributed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


## Citation
```
@article{10.1162/tacl_a_00446,
    author = {Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and Kryściński, Wojciech and Schoelkopf, Hailey and Kong, Riley and Tang, Xiangru and Mutuma, Mutethia and Rosand, Ben and Trindade, Isabel and Bandaru, Renusree and Cunningham, Jacob and Xiong, Caiming and Radev, Dragomir},
    title = "{FeTaQA: Free-form Table Question Answering}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {35-49},
    year = {2022},
    month = {01},
    abstract = "{Existing table question answering datasets contain abundant factual questions that primarily evaluate a QA system’s comprehension of query and tabular data. However, restricted by their short-form answers, these datasets fail to include question–answer interactions that represent more advanced and naturally occurring information needs: questions that ask for reasoning and integration of information pieces retrieved from a structured knowledge source. To complement the existing datasets and to reveal the challenging nature of the table-based question answering task, we introduce FeTaQA, a new dataset with 10K Wikipedia-based \\{table, question, free-form answer, supporting table cells\\} pairs. FeTaQA is collected from noteworthy descriptions of Wikipedia tables that contain information people tend to seek; generation of these descriptions requires advanced processing that
                    humans perform on a daily basis: Understand the question and table, retrieve, integrate, infer, and conduct text planning and surface realization to generate an answer. We provide two benchmark methods for the proposed task: a pipeline method based on semantic parsing-based QA systems and an end-to-end method based on large pretrained text generation models, and show that FeTaQA poses a challenge for both methods.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00446},
    url = {https://doi.org/10.1162/tacl\_a\_00446},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00446/1984125/tacl\_a\_00446.pdf},
}
```
