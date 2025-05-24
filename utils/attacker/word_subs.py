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


class WordSubsAttacker(BaseAttacker):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)

        if torch.cuda.is_available() and self.config_dict['use_cuda']:
            if self.config_dict['use_cuda_id'] is not None:
                self.device = torch.device(f"cuda:{self.config_dict['use_cuda_id']}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.word_subs_model = BertForMaskedLM.from_pretrained(
            self.config_dict['WordSubs_model'], output_hidden_states=True
        ).to(self.device)
        self.word_subs_tokenizer = BertTokenizer.from_pretrained(self.config_dict['WordSubs_model'])

        self.nlp = spacy.load(self.config_dict['spacy_model'])

    @property
    def attacker_method_name(self) -> str:
        return "WordSubs"

    def _candidates_gen(self, tokens, index_space: list[int], top_k: int = 8):
        text_tokens = [token.text for token in tokens]

        input_ids_bert = self.word_subs_tokenizer.convert_tokens_to_ids(text_tokens)

        # Create a tensor of input IDs
        input_tensor = torch.tensor([input_ids_bert]).to(self.device)

        with torch.no_grad():
            embeddings = self.word_subs_model.bert.embeddings(input_tensor.repeat(len(index_space), 1))

        with torch.no_grad():
            outputs = self.word_subs_model(inputs_embeds=embeddings)

        all_processed_tokens = []
        for i, masked_token_index in enumerate(index_space):
            if input_ids_bert[masked_token_index] in self.word_subs_tokenizer.all_special_ids:
                all_processed_tokens.append(text_tokens[masked_token_index])  # 是tokenizer定义的special_ids就用原词
            else:
                predicted_logits = outputs[0][i][masked_token_index]
                # Set the number of top predictions to return
                # Get the top n predicted tokens and their probabilities
                probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
                top_n_probs, top_n_indices = torch.topk(probs, top_k)
                top_n_tokens = self.word_subs_tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
                # print(top_n_tokens)
                if top_n_tokens[0] != text_tokens[masked_token_index]:
                    gold_subs = top_n_tokens[0]
                else:
                    gold_subs = top_n_tokens[1]
                all_processed_tokens.append(gold_subs)

        return all_processed_tokens

    def attack(self, *args, **kwargs) -> BaseAttackerReturn:
        pass
