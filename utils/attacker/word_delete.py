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

    def attack(
            self, text: str,
            attack_rate: float | list[float],
            start: int = 0, end: int = -1,
            seed: int = 2049
    ) -> BaseAttackerReturn:
        text_doc = self.nlp(text)

        if end <= 0:
            end = len(text_doc)

        if isinstance(attack_rate, float):
            attack_rate_list = [attack_rate]
        else:
            attack_rate_list = sorted(attack_rate)

        text_tokens = [token for token in text_doc]

        # 去除标点符号
        punct_index = punctuation_detect(text_tokens)
        # 去除数字
        number_index = number_detect(text_tokens)

        # 最大的攻击概率(从排序后的列表中取最后一个元素)
        attack_rate_max = attack_rate_list[-1]

        # 有效的tokens个数
        good_tokens_num = len(text_tokens) - len(punct_index) - len(number_index)

        # 根据最大攻击概率计算将要攻击的单词数
        num_max = int(attack_rate_max * good_tokens_num)
        # print(n)

        # 生成被攻击单词下标
        random_int_list: list[int] = generate_unique_random_integers(
            num_max, start, end, seed=seed, excluded_list=punct_index + number_index
        )

        words = [token.text for token in text_doc]
        corrupted_texts = []
        for rate in attack_rate_list:
            n = int(rate * good_tokens_num)
            temp_words = [word for i, word in enumerate(words) if i not in random_int_list[:n]]
            corrupted_text = " ".join(temp_words).strip()
            corrupted_texts.append(corrupted_text)

        return BaseAttackerReturn(
            text_original=text,
            text_attacked={
                'rate_list': attack_rate,
                'text_list': corrupted_texts,
            }
        )
