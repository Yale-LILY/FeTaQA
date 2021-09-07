#setup
import json
import logging
import os
import re
import sys

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np

def concat_linearize_table_context(table_array, question):
    return question+(' '.join([' '.join(row) for row in table_array]))


def default_linearize_table_context(table_array, question):
    return '[CLS]'+question+'[CLS]'+('[SEP]'.join([' '.join(row) for row in table_array]))

def sample_linearize_table_context(table_array, question):
    sample_len = 21
    simple_lin = '[CLS]'+question+'[CLS]'+('[SEP]'.join([' '.join(row) for row in table_array]))
    if len(table_array) >= sample_len:
        mask = np.zeros(len(table_array)-1, dtype=int)
        mask[:sample_len] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        lin = '[CLS]'+question+'[CLS]'+ ' '.join(table_array[0])
        sampled_rows = [row for boo, row in zip(mask,table_array[1:]) if boo]
        lin += '[SEP]'.join([' '.join(row) for row in sampled_rows])
        return lin
        
    return simple_lin

linearization_dic = {
    'simple':default_linearize_table_context,
    'sample':sample_linearize_table_context,
    'concat':concat_linearize_table_context
}
def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)
