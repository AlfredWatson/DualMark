import random
import copy

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy

from utils.util_str import StrFactory
from utils.base_attacker import BaseAttacker, BaseAttackerReturn
from exceptions.exceptions import DiversityValueError


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


class DipperAttacker(BaseAttacker):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)

        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained(self.config_dict['dipper_tokenizer'])
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained(
            self.config_dict['dipper_model'], device_map='auto'
        )
        self.gen_kwargs = self.config_dict['generate_kwargs']

        self.nlp = spacy.load(self.config_dict['spacy_model'])

        self.lex_diversity = self.config_dict['lexical_diversity']
        self.order_diversity = self.config_dict['order_diversity']
        self.sent_interval = self.config_dict['sentence_interval']

    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity-value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    @property
    def attacker_method_name(self) -> str:
        return "Dipper"

    def attack(self, *args, **kwargs) -> BaseAttackerReturn:
        pass
