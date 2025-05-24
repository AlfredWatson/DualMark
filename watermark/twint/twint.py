from math import sqrt

import torch
from transformers import LogitsProcessor

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from watermark.upv.network_model import UPVGenerator, UPVDetector


class TwintConfig(BaseConfig):
    """Config class for KGW+UPV algorithm"""

    def __init__(
            self, algorithm_config_path: str,
            transformers_config: TransformersConfig,
            *args, **kwargs
    ):
        super().__init__(algorithm_config_path, transformers_config, *args, **kwargs)

        self.lo_kgw = -2
        self.lo_upv = -1

        """Initialize KGW-specific parameters."""
        self.gamma_kgw = self.config_dict['gamma_1']
        self.delta_kgw = self.config_dict['delta_1']
        self.prefix_length_kgw = self.config_dict['prefix_length_1']
        self.z_threshold_kgw = self.config_dict['z_threshold_1']

        self.hash_key_kgw = self.config_dict['hash_key_1']
        self.f_scheme = self.config_dict['f_scheme_1']
        self.window_scheme = self.config_dict['window_scheme_1']

        """Initialize UPV-specific parameters."""
        self.gamma_upv = self.config_dict['gamma_2']
        self.delta_upv = self.config_dict['delta_2']
        self.prefix_length_upv = self.config_dict['prefix_length_2']
        self.z_threshold_upv = self.config_dict['z_threshold_2']

        self.bit_number_upv = self.config_dict['bit_number']
        self.sigma_upv = self.config_dict['sigma']
        self.default_top_k_upv = self.config_dict['default_top_k']
        self.generator_model_name = self.config_dict['generator_model_name']
        self.detector_model_name = self.config_dict['detector_model_name']
        self.detect_mode_upv = self.config_dict['detect_mode']

    def initialize_parameters(self) -> None:
        print("Initialize Twint(KGW+UPV)-specific parameters.")

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Twint'


class TwintUtils:

    def __init__(self, config: TwintConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng_1 = torch.Generator(device=self.config.device)
        self.rng_1.manual_seed(self.config.hash_key_kgw)
        self.prf_1 = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng_1)
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min
        }
        self.window_scheme_map = {"left": self._get_greenlist_ids_left_1, "self": self._get_greenlist_ids_self_1}

        self.config = config
        self.generator_model_2 = self._get_generator_model_2(
            self.config.bit_number_upv, self.config.prefix_length_upv + 1
        ).to(self.config.device)
        self.detector_model_2 = self._get_detector_model_2(self.config.bit_number_upv).to(self.config.device)
        self.cache_2 = {}
        self.top_k_2 = self.config.gen_kwargs.get('top_k', self.config.default_top_k_upv)
        self.num_beams_2 = self.config.gen_kwargs.get('num_beams', None)

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
        """Get green-list ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left_1(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get green-list ids for the input_ids via leftHash scheme."""
        self.rng_1.manual_seed(self.config.hash_key_kgw * self._f(input_ids))
        greenlist_size = int(self.config.vocab_size * self.config.gamma_kgw)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng_1)
        green_list_ids = vocab_permutation[:greenlist_size]
        return green_list_ids

    def _get_greenlist_ids_self_1(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get green-list ids for the input_ids via selfHash scheme."""
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

    def _compute_z_score_kgw(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma_kgw
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
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

        z_score = self._compute_z_score_kgw(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    def _get_generator_model_2(self, input_dim: int, window_size: int) -> UPVGenerator:
        """Load the generator model from the specified file."""
        model = UPVGenerator(input_dim, window_size)
        model.load_state_dict(torch.load(self.config.generator_model_name))
        return model

    def _get_detector_model_2(self, bit_number: int) -> UPVDetector:
        """Load the detector model from the specified file."""
        model = UPVDetector(bit_number)
        model.load_state_dict(torch.load(self.config.detector_model_name))
        return model

    def _get_predictions_from_generator_2(self, input_x: torch.Tensor) -> bool:
        """Get predictions from the generator model."""
        with torch.no_grad():
            output = self.generator_model_2(input_x)
            output = (output > 0.5).bool().item()
        return output

    def int_to_bin_list_2(self, n: int, length=8) -> list[int]:
        """Convert an integer to a binary list of specified length."""
        bin_str = format(n, 'b')[:length].zfill(length)
        return [int(b) for b in bin_str]

    def _select_candidates_2(self, scores: torch.Tensor) -> torch.Tensor:
        """Select candidate tokens based on the scores."""
        if self.num_beams_2 is not None:
            threshold_score = torch.topk(scores, self.num_beams_2, largest=True, sorted=False)[0][-1]
            return (scores >= (threshold_score - self.config.delta_upv)).nonzero(as_tuple=True)[0]
        else:
            return torch.topk(scores, self.top_k_2, largest=True, sorted=False).indices

    def get_greenlist_ids_upv(self, input_ids: torch.Tensor, scores: torch.Tensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        greenlist_ids = []
        candidate_tokens = self._select_candidates_2(scores)

        # Ensure input_ids is a list for concatenation
        input_ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

        for v in candidate_tokens:
            # Now safely concatenate lists
            pair = (input_ids_list[-self.config.prefix_length_upv:]
                    + [v.item()]) if self.config.prefix_length_upv > 0 else [v.item()]
            merged_tuple = tuple(pair)

            if merged_tuple in self.cache_2:
                result = self.cache_2[merged_tuple]
            else:
                bin_list = [self.int_to_bin_list_2(num, self.config.bit_number_upv) for num in pair]
                result = self._get_predictions_from_generator_2(
                    torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
                )
                self.cache_2[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids

    def _judge_green_2(
            self, input_ids: torch.Tensor,
            current_number: int
    ) -> bool:
        """Judge if the current token is green based on previous tokens."""

        # Get the last 'prefix_length' items from input_ids
        last_nums = input_ids[-self.config.prefix_length_upv:] if self.config.prefix_length_upv > 0 else []
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [self.int_to_bin_list_2(num, self.config.bit_number_upv) for num in pair]

        # load & update cache
        if merged_tuple in self.cache_2:
            result = self.cache_2[merged_tuple]
        else:
            result = self._get_predictions_from_generator_2(
                torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
            )
            self.cache_2[merged_tuple] = result

        return result

    def green_token_mask_and_stats_2(self, input_ids: torch.Tensor) -> tuple[list[bool], int, float]:
        """Get green token mask and statistics for the input_ids."""

        # Initialize a list with None for the prefix tokens which are not scored
        mask_list = [-1] * self.config.prefix_length_upv

        # Count of green tokens, initialized to zero
        green_token_count = 0

        # Iterate over each token in the input_ids starting from prefix_length
        for idx in range(self.config.prefix_length_upv, len(input_ids)):
            # Get the current token
            curr_token = input_ids[idx]

            # Judge if the current token is green based on previous tokens
            if self._judge_green_2(input_ids[:idx], curr_token.item()):
                mask_list.append(1)  # Mark this token as green
                green_token_count += 1  # Increment the green token counter
            else:
                mask_list.append(0)  # Mark this token as not green

        # Compute the number of tokens that were evaluated for green status
        num_tokens_scored = len(input_ids) - self.config.prefix_length_upv

        # Calculate the z-score for the number of green tokens
        z_score = self._compute_z_score_2(green_token_count, num_tokens_scored)

        # Return the mask list, count of green tokens, and the z-score
        return mask_list, green_token_count, z_score

    def _compute_z_score_2(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma_upv
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count) + self.config.sigma_upv * self.config.sigma_upv * T)
        z = numer / denom
        return z


class TwintLogitsProcessor(LogitsProcessor):

    def __init__(self, config: TwintConfig, utils: TwintUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def _bias_greenlist_logits(
            self, scores: torch.Tensor,
            greenlist_mask: torch.Tensor,
            greenlist_bias: float
    ) -> torch.Tensor:
        """Bias the scores for the green-list tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _calc_greenlist_mask(
            self, scores: torch.Tensor,
            greenlist_token_ids: torch.Tensor
    ) -> torch.Tensor:
        green_tokens_mask = torch.zeros_like(scores)
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

        for batch_idx in range(input_ids.shape[0]):
            greenlist_ids_kgw = self.utils.get_greenlist_ids_kgw(input_ids[batch_idx])
            greenlist_ids_upv = self.utils.get_greenlist_ids_upv(input_ids[batch_idx], scores=scores[batch_idx])

            greenlist_ids_kgw_set = set(greenlist_ids_kgw.tolist())
            greenlist_ids_upv_set = set(greenlist_ids_upv)
            common_set = greenlist_ids_kgw_set & greenlist_ids_upv_set
            common_ids = torch.tensor(list(common_set), dtype=torch.long)

            green_tokens_mask = self._calc_greenlist_mask(
                scores=scores[batch_idx], greenlist_token_ids=common_ids
            )
            scores[batch_idx][green_tokens_mask] = (
                scores[batch_idx][green_tokens_mask]
                + self.config.delta_kgw
                + self.config.delta_upv
            )
            
        return scores


class Twint(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: TwintConfig) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (TwintConfig): KGWConfig instance.
        """
        utils = TwintUtils(algorithm_config)
        logits_processor = TwintLogitsProcessor(algorithm_config, utils)
        super().__init__(algorithm_config, utils, logits_processor)
        self.config = algorithm_config
        self.utils = utils
        self.logits_processor = logits_processor

    def _detect_watermark_network_mode(self, encoded_text: torch.Tensor) -> tuple[bool, None]:
        """ Detect watermark using the network mode. """
        # Convert input IDs to binary sequence
        inputs_bin = [self.utils.int_to_bin_list_2(token_id, self.config.bit_number_upv) for token_id in encoded_text]
        inputs_bin = torch.tensor(inputs_bin, device=self.config.device)

        # Run the model on the input binary sequence
        outputs = self.utils.detector_model_2(inputs_bin.unsqueeze(dim=0).float())
        outputs = outputs.reshape([-1])
        predicted = (outputs.data > 0.5).int()

        # Determine watermark presence based on predictions
        is_watermarked = (predicted == 1).sum().item() > 0

        # z_score is not applicable in network mode
        return is_watermarked, None

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        return super.detect_watermark(text, return_dict,  *args, **kwargs)
