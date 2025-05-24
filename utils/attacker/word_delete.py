import copy
import random

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils.base_attacker import BaseAttacker, BaseAttackerReturn


def punctuation_detect(doc_tokens):
    """
    返回句子中所有标点符号的下标
    Args:
        doc_tokens: 经过spaCy处理过的文本的Token list
    Returns: 所有标点符号的下标list
    """
    punct_index = []
    for i, token in enumerate(doc_tokens):
        if token.is_punct:
            punct_index.append(i)
    return punct_index


def number_detect(doc_tokens):
    number_index = []
    for i, token in enumerate(doc_tokens):
        s = token.text
        for c in s:
            if c.isdigit():
                number_index.append(i)
                break
    return number_index


def generate_unique_random_integers(n, min, max, seed=None, excluded_list=None):
    """
    在[min,max]内生成n个整数，且这n个数字不在已给出的数字列表excluded_list中
    Args:
        n: 个数
        min: 最小
        max: 最大
        seed: 随机种子
        excluded_list: 剔除表
    Returns: n个不重复的整数s
    """
    if n > (max - min + 1):  # 确保要生成的数量不超过范围内的整数总数
        raise ValueError("Cannot generate more unique integers than the range size.")

    if excluded_list is not None:
        all_numbers = list(set(range(min, max)) - set(excluded_list))
    else:
        all_numbers = list(set(range(min, max)))

    random.seed(seed)
    return random.sample(all_numbers, n)


class WordDeleteAttacker(BaseAttacker):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)

        self.nlp = spacy.load(self.config_dict['spacy_model'])

    @property
    def attacker_method_name(self) -> str:
        return "WordDelete"

    def attack(self, *args, **kwargs) -> BaseAttackerReturn:
        pass
