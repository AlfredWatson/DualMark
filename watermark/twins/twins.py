import math

import torch
from transformers import AutoTokenizer
from transformers import LogitsProcessor
from transformers import OPTForCausalLM

from ..base import BaseWatermark, BaseConfig
from ..ts.TS_networks import DeltaNetwork, GammaNetwork
from utils.transformers_config import TransformersConfig


class TwinsConfig(BaseConfig):
    """Config class for KGW+TS algorithm, load config file and initialize parameters."""

    def __init__(
            self, algorithm_config_path: str,
            transformers_config: TransformersConfig,
            *args, **kwargs
    ):
        super().__init__(algorithm_config_path, transformers_config, *args, **kwargs)

        self.lo_kgw = -2
        self.lo_ts = -1

        """Initialize KGW-specific parameters."""
        self.gamma_kgw = self.config_dict['gamma_1']
        self.delta_kgw = self.config_dict['delta_1']
        self.hash_key_kgw = self.config_dict['hash_key_1']
        self.z_threshold_kgw = self.config_dict['z_threshold_1']
        self.prefix_length_kgw = self.config_dict['prefix_length_1']

        self.f_scheme = self.config_dict['f_scheme_1']
        self.window_scheme = self.config_dict['window_scheme_1']

        """Initialize TS-specific parameters."""
        self.generation_model_name: str = self.generation_model.model.name_or_path

        self.gamma_ts = self.config_dict['gamma_2']
        self.delta_ts = self.config_dict['delta_2']
        self.prefix_length_ts = self.config_dict['prefix_length_2']
        self.z_threshold_ts = self.config_dict['z_threshold_2']
        self.hash_key_ts = self.config_dict['hash_key_2']

        self.seeding_scheme = self.config_dict['seeding_scheme_2']
        self.ckpt_path = self.config_dict['ckpt_path_2']

        if "opt" not in self.generation_model_name.lower():
            self.tokenizer_opt = AutoTokenizer.from_pretrained(
                "/home/haojifei/develop_tools/transformers/models/facebook/opt-1.3b",
                padding_side="left"
            )
        else:
            self.tokenizer_opt = self.generation_tokenizer
        self.tokenizer_llama = self.generation_tokenizer

    def initialize_parameters(self) -> None:
        print("Initialize Twins(KGW+TS)-specific parameters.")

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Twins'


class TwinsUtils:

    def __init__(self, config: TwinsConfig) -> None:
        self.config = config

        self.rng_1 = torch.Generator(device=self.config.device)
        self.rng_1.manual_seed(self.config.hash_key_kgw)
        self.prf_1 = torch.randperm(
            self.config.vocab_size,
            device=self.config.device,
            generator=self.rng_1
        )
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min
        }
        self.window_scheme_map = {"left": self._get_greenlist_ids_left_1, "self": self._get_greenlist_ids_self_1}

        """Initialize the TS utility."""
        self.rng_2 = torch.Generator(device=self.config.device)
        self.rng_2.manual_seed(self.config.hash_key_ts)

        if "opt" not in self.config.generation_model_name.lower():
            temp_model = OPTForCausalLM.from_pretrained(
                "/home/haojifei/develop_tools/transformers/models/facebook/opt-1.3b",
                torch_dtype=torch.float,
                device_map='auto'
            )
            self.embed_matrix = temp_model.get_input_embeddings().weight.to(self.config.device)
            del temp_model
        else:
            temp_model = self.config.generation_model
            self.embed_matrix = temp_model.get_input_embeddings().weight.to(self.config.device)

        if len(self.config.ckpt_path) > 0:
            print("checkpoint_load")
            checkpoint = torch.load(self.config.ckpt_path)
            layer_delta = sum(
                1 for key in checkpoint['delta_state_dict'] if "weight" in key)  # Counting only weight keys as layers
            layer_gamma = sum(
                1 for key in checkpoint['gamma_state_dict'] if "weight" in key)  # Counting only weight keys as layers

            self.gamma_network_2 = GammaNetwork(
                input_dim=self.embed_matrix.shape[1], layers=layer_gamma
            ).to(self.config.device)
            self.delta_network_2 = DeltaNetwork(
                input_dim=self.embed_matrix.shape[1], layers=layer_delta
            ).to(self.config.device)

            self.delta_network_2.load_state_dict(checkpoint['delta_state_dict'])
            self.gamma_network_2.load_state_dict(checkpoint['gamma_state_dict'])

            for name, param in self.delta_network_2.named_parameters():
                param.requires_grad = False
            for name, param in self.gamma_network_2.named_parameters():
                param.requires_grad = False
            self.delta_network_2.eval()
            self.gamma_network_2.eval()
        else:
            self.gamma_2 = torch.tensor([self.config.gamma_ts]).to(self.config.device)
            self.delta_2 = torch.tensor([self.config.delta_ts]).to(self.config.device)

        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.config.device)

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length_kgw):
            time_result *= input_ids[self.config.lo_kgw - i].item()  # todo: -1
        return self.prf_1[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length_kgw):
            additive_result += input_ids[-1 - i].item()
        return self.prf_1[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf_1[input_ids[- self.config.prefix_length_kgw].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf_1[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length_kgw))

    def get_greenlist_ids_kgw(self, input_ids: torch.LongTensor | torch.cuda.LongTensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left_1(self, input_ids: torch.LongTensor | torch.cuda.LongTensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        self.rng_1.manual_seed(self.config.hash_key_kgw * self._f(input_ids))
        greenlist_size = int(self.config.vocab_size * self.config.gamma_kgw)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng_1)
        green_list_ids = vocab_permutation[:greenlist_size]
        return green_list_ids

    def _get_greenlist_ids_self_1(self, input_ids: torch.LongTensor | torch.cuda.LongTensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma_kgw)
        greenlist_ids = []
        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf_1[k])
            self.rng_1.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng_1)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return torch.tensor(greenlist_ids)

    def compute_z_score_kgw(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        numer = observed_count - self.config.gamma_kgw * T
        denom = math.sqrt(T * self.config.gamma_kgw * (1 - self.config.gamma_kgw))
        z = numer / denom
        return z

    def score_sequence_kgw(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length_kgw
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length_kgw} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length_kgw)]

        for idx in range(-1 * self.config.lo_kgw, len(input_ids) + self.config.lo_kgw):
            curr_token = input_ids[idx]
            green_list_ids = self.get_greenlist_ids_kgw(input_ids[:idx])
            if curr_token in green_list_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        z_score = self.compute_z_score_kgw(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    def _seed_rng_2(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.config.seeding_scheme

        # using the last token and hash_key to generate random seed
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, \
                f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[self.config.lo_ts].item()
            self.rng_2.manual_seed(self.config.hash_key_ts * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def get_green_list_ids_ts(
            self,
            input_ids: torch.LongTensor | torch.cuda.LongTensor,
            process: str
    ):
        # Always use ids given by OPT model
        # since our gamma/delta network is trained on the embedding matrix of OPT model
        # seed the rng using the previous tokens/prefix according to the seeding_scheme
        self._seed_rng_2(input_ids)

        # use last token to get gamma value and delta value
        if len(self.config.ckpt_path) > 0:
            gamma = self.gamma_network_2(self.embed_matrix[input_ids[self.config.lo_ts].item()])
            delta = self.delta_network_2(self.embed_matrix[input_ids[self.config.lo_ts].item()])
        else:
            delta = self.delta_2
            gamma = self.gamma_2

        if process == 'process':
            # get every token's gamma value and delta value
            self.gamma_list = torch.cat([self.gamma_list, gamma])
            self.delta_list = torch.cat([self.delta_list, delta])
        else:
            self.gamma_list = torch.cat([self.gamma_list, gamma])
            self.delta_list = torch.cat([self.delta_list, delta])

        # generate greenlist, every token have different greenlist_id
        green_list_size = int(self.config.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng_2)

        green_list_ids = vocab_permutation[:green_list_size]

        gamma_len = len(self.gamma_list)

        return green_list_ids, gamma, delta, gamma_len

    def compute_z_score_ts(self, observed_count: int, T: int) -> float:
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list)).item()
        mean = torch.sum(self.gamma_list).item()
        z = (observed_count - mean) / math.sqrt(var)
        return z

    def compute_z_score_2_1(self, observed_count: int) -> float:
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list)).item()
        mean = torch.sum(self.gamma_list).item()
        z = (observed_count - mean) / math.sqrt(var)
        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        return z

    def score_sequence_ts(self, input_ids: torch.Tensor, ) -> tuple[float, list[int]]:

        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.config.device)

        num_tokens_scored = len(input_ids) - self.config.prefix_length_ts
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length_ts} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_mask = [-1 for _ in range(self.config.prefix_length_ts)]

        for idx in range(self.config.prefix_length_ts, len(input_ids)):
            curr_token = input_ids[idx]
            if "opt" not in self.config.generation_model_name.lower():
                llama_str = self.config.tokenizer_llama.decode(input_ids[max(idx - 5, 0):idx], add_special_tokens=False)
                ids_opt = self.config.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
                if len(ids_opt) == 0:
                    green_token_mask.append(0)
                    continue
                green_list_ids, gamma, delta, gamma_len = self.get_green_list_ids_ts(
                    torch.tensor(ids_opt).to(self.config.device),
                    "detect"
                )
            else:
                green_list_ids, gamma, delta, gamma_len = self.get_green_list_ids_ts(
                    input_ids[:idx], "detect"
                )

            if curr_token in green_list_ids:
                green_token_count += 1
                green_token_mask.append(1)
            else:
                green_token_mask.append(0)

        self.gamma_list = self.gamma_list[self.config.prefix_length_ts:]

        z_score = self.compute_z_score_ts(green_token_count, num_tokens_scored)

        return z_score, green_token_mask


class TwinsLogitsProcessor(LogitsProcessor):

    def __init__(self, config: TwinsConfig, utils: TwinsUtils) -> None:
        self.config = config
        self.utils = utils

    def compute_entropy(self, prob_distribution):
        """计算给定概率分布的熵"""
        entropy = -torch.sum(prob_distribution * torch.log2(prob_distribution + 1e-10), dim=-1)
        return entropy

    def _calc_greenlist_mask(
            self, logits: torch.Tensor,
            greenlist_token_ids: torch.Tensor
    ) -> torch.Tensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(
            self, input_ids: torch.LongTensor,
            scores: torch.FloatTensor
    ) -> torch.FloatTensor | torch.cuda.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length_kgw:
            return scores

        if "opt" not in self.config.generation_model_name.lower():
            llama_str = self.config.tokenizer_llama.batch_decode(input_ids[:, -5:], add_special_tokens=False)
            input_ids_opt = self.config.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
            input_ids_ts = torch.tensor(input_ids_opt)
        else:
            input_ids_ts = input_ids

        for batch_idx in range(input_ids.shape[0]):
            green_list_ids_kgw = self.utils.get_greenlist_ids_kgw(input_ids[batch_idx])
            # green_tokens_mask = self._calc_greenlist_mask(
            #     logits=scores[batch_idx], greenlist_token_ids=green_list_ids_kgw
            # )
            # scores[batch_idx][green_tokens_mask] = scores[batch_idx][green_tokens_mask] + self.config.delta_kgw

            green_list_ids_ts, gamma_ts, delta_ts, gamma_len_ts = self.utils.get_green_list_ids_ts(
                input_ids_ts[batch_idx], 'process'
            )
            # green_tokens_mask = self._calc_greenlist_mask(
            #     logits=scores[batch_idx], greenlist_token_ids=green_list_ids_ts
            # )
            # delta_ts = delta_ts.to(dtype=scores.dtype)
            # scores[batch_idx][green_tokens_mask] = scores[batch_idx][green_tokens_mask] + delta_ts

            greenlist_ids_1_set = set(green_list_ids_kgw.tolist())
            greenlist_ids_2_set = set(green_list_ids_ts.tolist())
            common_set = greenlist_ids_1_set & greenlist_ids_2_set
            common_ids = torch.tensor(list(common_set), dtype=torch.long)

            # get green-list token mask and add bias on each logits base on green-list mask
            green_tokens_mask = self._calc_greenlist_mask(
                logits=scores[batch_idx], greenlist_token_ids=common_ids
            )

            delta_ts = delta_ts.to(dtype=scores.dtype)
            scores[batch_idx][green_tokens_mask] = (scores[batch_idx][green_tokens_mask]
                                                    + delta_ts
                                                    + self.config.delta_kgw)

            # 将 logits 转换为概率分布
            # probs = torch.softmax(scores[batch_idx], dim=-1)
            # 提取目标位置的真实 token
            # target_tokens = input_ids[:, 1:]  # 去掉第一个 token（<s>）
            # predicted_probs = probs[:, :-1].gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            # 计算每个位置的熵
            # entropy_per_token = -torch.log2(predicted_probs + 1e-10)  # 防止 log(0)
            # average_entropy = entropy_per_token.mean().item()

        return scores


class Twins(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: TwinsConfig) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (TwinsConfig): KGWConfig instance.
        """
        utils = TwinsUtils(algorithm_config)
        logits_processor = TwinsLogitsProcessor(algorithm_config, utils)
        super().__init__(algorithm_config, utils, logits_processor)

        self.config = algorithm_config
        self.utils = utils
        self.logits_processor = logits_processor

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)
        # haojifei = min(self.config.lo_ts, self.config.lo_kgw)
        # encoded_text = encoded_text[self.config.gen_kwargs['min_length'] - self.config.gen_kwargs['max_new_tokens']
        #                             + haojifei - 1:]

        # Compute z_score using a utility method
        z_score_1, green_token_flags_1 = self.utils.score_sequence_kgw(encoded_text)
        z_score_2, green_token_flags_2 = self.utils.score_sequence_ts(encoded_text)

        ts = self.config.generation_tokenizer.convert_ids_to_tokens(encoded_text)
        for m1, m2, t in zip(green_token_flags_1, green_token_flags_2, ts):
            print(f"{t}: {m1},{m2}")
            if m1 == 1 and m2 == 1:
                print("dual", "===" * 3)
            if m1 == 1 and m2 == 0:
                print("kgw", "===" * 3)
            if m1 == 0 and m2 == 1:
                print("tsw", "===" * 3)

        # green_token_flags_union = []
        # green_token_count = 0
        # for f1, f2 in zip(green_token_flags_1, green_token_flags_2):
        #     if f1 == 0 and f2 == 0:
        #         green_token_flags_union.append(0)
        #     else:
        #         green_token_count += 1
        #         green_token_flags_union.append(1)

        # num_tokens_scored = len(encoded_text) - self.config.prefix_length_1
        # z_score_1 = self.utils.compute_z_score_1(green_token_count, num_tokens_scored)
        # z_score_2 = self.utils.compute_z_score_2_1(green_token_count)

        # Determine if the z_score indicates a watermark
        is_watermarked_1: bool = z_score_1 > self.config.z_threshold_kgw
        is_watermarked_2: bool = z_score_2 > self.config.z_threshold_ts

        is_watermarked: bool = True if is_watermarked_1 or is_watermarked_2 else False

        # Return results based on the return_dict flag
        if return_dict:
            return {
                "is_watermarked": is_watermarked,
                "score": (z_score_1 + z_score_2) / 2,
                "is_watermarked1": is_watermarked_1,
                "score1": z_score_1,
                "is_watermarked2": is_watermarked_2,
                "score2": z_score_2,
            }
        else:
            return is_watermarked_1, z_score_1, is_watermarked_2, z_score_2
