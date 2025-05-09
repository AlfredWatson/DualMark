import math

import torch
from transformers import LogitsProcessor
from transformers import OPTForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.base import BaseWatermark, BaseConfig
from watermark.ts.TS_networks import DeltaNetwork, GammaNetwork
from visualize.data_for_visualization import DataForVisualization


class TSConfig(BaseConfig):

    def __init__(
            self, algorithm_config_path: str,
            transformers_config: TransformersConfig,
            *args, **kwargs
    ):

        super().__init__(algorithm_config_path, transformers_config, *args, **kwargs)
        self.generation_model_name: str = self.generation_model.model.name_or_path()

        """Initialize algorithm-specific parameters."""
        self.gamma_ts = self.config_dict['gamma']
        self.delta_ts = self.config_dict['delta']
        self.hash_key_ts = self.config_dict['hash_key']
        self.prefix_length_ts = self.config_dict['prefix_length']
        self.z_threshold_ts = self.config_dict['z_threshold']

        self.seeding_scheme = self.config_dict['seeding_scheme']
        self.ckpt_path = self.config_dict['ckpt_path']


        if "opt" not in self.generation_model_name.lower():
            self.tokenizer_opt = AutoTokenizer.from_pretrained(
                "/home/haojifei/develop_tools/transformers/models/facebook/opt-1.3b",
                padding_side="left"
            )
        else:
            self.tokenizer_opt = self.generation_tokenizer
        self.tokenizer_llama = self.generation_tokenizer

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'TS'


class TSUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: TSConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config
        self.rng_ts = torch.Generator(device=self.config.device)
        self.rng_ts.manual_seed(self.config.hash_key_ts)

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
            # Counting only weight keys as layers
            layer_delta = sum(1 for key in checkpoint['delta_state_dict'] if "weight" in key)
            layer_gamma = sum(1 for key in checkpoint['gamma_state_dict'] if "weight" in key)

            self.gamma_network = GammaNetwork(
                input_dim=self.embed_matrix.shape[1], layers=layer_gamma
            ).to(self.config.device)
            self.delta_network = DeltaNetwork(
                input_dim=self.embed_matrix.shape[1], layers=layer_delta
            ).to(self.config.device)

            self.delta_network.load_state_dict(checkpoint['delta_state_dict'])
            self.gamma_network.load_state_dict(checkpoint['gamma_state_dict'])

            for name, param in self.delta_network.named_parameters():
                param.requires_grad = False
            for name, param in self.gamma_network.named_parameters():
                param.requires_grad = False
            self.delta_network.eval()
            self.gamma_network.eval()

        else:
            self.gamma = torch.tensor([self.config.gamma_ts]).to(self.config.device)
            self.delta = torch.tensor([self.config.delta_ts]).to(self.config.device)

        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.config.device)

    def _seed_rng_ts(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.config.seeding_scheme

        # using the last token and hash_key to generate random seed
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, \
                f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng_ts.manual_seed(self.config.hash_key_ts * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def get_green_list_ids_ts(
            self, input_ids: torch.LongTensor | torch.cuda.LongTensor,
            process: str
    ):
        # Always use ids given by OPT model
        # since our gamma/delta network is trained on the embedding matrix of OPT model
        # seed the rng using the previous tokens/prefix according to the seeding_scheme
        self._seed_rng_ts(input_ids)

        # use last token to get gamma value and delta value
        if len(self.config.ckpt_path) > 0:
            gamma = self.gamma_network(self.embed_matrix[input_ids[-1].item()])
            delta = self.delta_network(self.embed_matrix[input_ids[-1].item()])
        else:
            delta = self.delta
            gamma = self.gamma

        if process == 'process':
            # get every token's gamma value and delta value
            self.gamma_list = torch.cat([self.gamma_list, gamma])
            self.delta_list = torch.cat([self.delta_list, delta])
        else:  # todo: 后加
            self.gamma_list = torch.cat([self.gamma_list, gamma])
            self.delta_list = torch.cat([self.delta_list, delta])

        # generate greenlist, every token have different greenlist_id
        green_list_size = int(self.config.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng_ts)

        green_list_ids = vocab_permutation[:green_list_size]

        gamma_len = len(self.gamma_list)

        return green_list_ids, gamma, delta, gamma_len

    def _compute_z_score_ts(self, observed_count: int, T: int) -> float:
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list)).item()
        mean = torch.sum(self.gamma_list).item()
        z = (observed_count - mean) / math.sqrt(var)
        return z

    def score_sequence_ts(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""

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
                    green_token_mask.append(False)
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

        z_score = self._compute_z_score_ts(green_token_count, num_tokens_scored)

        # todo: 后加: 清空
        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.config.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.config.device)

        return z_score, green_token_mask


class TSLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for TS algorithm, process logits to add watermark."""

    def __init__(self, config: TSConfig, utils: TSUtils, *args, **kwargs) -> None:

        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(
            self,
            logits: torch.FloatTensor | torch.cuda.FloatTensor,
            greenlist_token_ids: torch.LongTensor | torch.cuda.LongTensor
    ) -> torch.BoolTensor | torch.cuda.BoolTensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.utils.rng_ts is None:
            self.utils.rng_ts = torch.Generator(device=self.config.device)

        if "opt" not in self.config.generation_model_name.lower():
            llama_str = self.config.tokenizer_llama.batch_decode(input_ids[:, -5:], add_special_tokens=False)
            input_ids_opt = self.config.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
            del input_ids
            input_ids = torch.tensor(input_ids_opt).to(self.config.device)

        for b_idx in range(input_ids.shape[0]):
            green_list_ids, gamma, delta, gamma_len = self.utils.get_green_list_ids_ts(
                input_ids[b_idx], 'process'
            )

            # get green-list token mask and add bias on each logits base on green-list mask
            green_tokens_mask = self._calc_greenlist_mask(
                logits=scores[b_idx], greenlist_token_ids=green_list_ids
            )

            delta = delta.to(dtype=scores.dtype)
            scores[b_idx][green_tokens_mask] = scores[b_idx][green_tokens_mask] + delta

        return scores


class TS(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: TSConfig) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (TSConfig): TSConfig instance.
        """
        utils = TSUtils(algorithm_config)
        logits_processor = TSLogitsProcessor(algorithm_config, utils)
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

        # Compute z_score using a utility method
        z_score, _ = self.utils.score_sequence_ts(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked: bool = z_score > self.config.z_threshold_ts

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return is_watermarked, z_score
