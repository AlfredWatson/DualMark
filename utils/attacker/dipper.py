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

    def attack(
            self, text: str,
            prompt: str,
            attack_rate: float | list[float],
    ) -> BaseAttackerReturn:

        # 规范攻击概率列表
        if isinstance(attack_rate, float):
            attack_rate_list = [attack_rate]
        else:
            attack_rate_list = sorted(attack_rate)

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)

        str_factory = StrFactory('en', self.nlp)

        # Preprocess the reference text
        prefix = str_factory.standard_str(prompt)

        text_doc = self.nlp(text)
        sentences = [sent.text for sent in text_doc.sents]
        for i, sent in enumerate(sentences):
            sentences[i] = str_factory.standard_str(sent)

        # texts = []
        # for r in attack_rate_list:
        #     n = int(r * len(sentences))
        #     texts.append(sentences[:n])

        sentences_dipper = []
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])

            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            # Tokenize the input
            final_input = self.paraphrase_tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            # Generate the edited text
            with torch.inference_mode():
                outputs = self.paraphrase_model.generate(**final_input, **self.gen_kwargs)
            outputs = self.paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Update the prefix and output text
            prefix += " " + outputs[0]
            sentences_dipper.append(outputs[0])

        random_int_list: list[int] = generate_unique_random_integers(
            int(attack_rate_list[-1] * len(sentences)), 0, len(sentences)
        )

        corrupted_texts = []
        for r in attack_rate_list:
            temp_sentences = copy.deepcopy(sentences)
            n = int(r * len(sentences))
            for i in random_int_list[:n]:
                temp_sentences[i] = sentences_dipper[i]
            temp_corrupted_text = " ".join(temp_sentences)
            corrupted_texts.append(' ' + temp_corrupted_text)

        for i, c in enumerate(corrupted_texts):
            # 将句子标记化为 token IDs
            encoded = self.paraphrase_tokenizer(c, return_tensors="pt")
            input_ids = encoded["input_ids"][0]
            # 去除前 30 个 token
            if len(input_ids) > 30:
                truncated_input_ids = input_ids[30:]
            else:
                truncated_input_ids = input_ids
            truncated_sentence = self.paraphrase_tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
            corrupted_texts[i] = truncated_sentence

        return BaseAttackerReturn(
            text_original=text,
            text_attacked={
                'rate_list': attack_rate_list,
                'text_list': corrupted_texts,
            }
        )
