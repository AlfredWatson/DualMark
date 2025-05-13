import math

import torch
from transformers import LogitsProcessor
from transformers import OPTForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.base import BaseWatermark, BaseConfig
from watermark.ts.TS_networks import DeltaNetwork, GammaNetwork
from watermark.upv.network_model import UPVGenerator, UPVDetector


class TwinuConfig(BaseConfig):
    """Config class for TS+UPV algorithm, load config file and initialize parameters."""

    def __init__(
            self, algorithm_config_path: str,
            transformers_config: TransformersConfig,
            *args, **kwargs
    ):
        super().__init__(algorithm_config_path, transformers_config, *args, **kwargs)

        self.lo_ts = -2
        self.lo_upv = -1

        """Initialize TS-specific parameters."""
        self.generation_model_name: str = self.generation_model.model.name_or_path
        self.gamma_ts = self.config_dict['gamma_2']
        self.delta_ts = self.config_dict['delta_2']
        self.hash_key_ts = self.config_dict['hash_key']
        self.prefix_length_ts = self.config_dict['prefix_length_2']
        self.z_threshold_ts = self.config_dict['z_threshold_2']

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

        """Initialize UPV-specific parameters."""
        self.gamma_upv = self.config_dict['gamma_1']
        self.delta_upv = self.config_dict['delta_1']
        self.z_threshold_upv = self.config_dict['z_threshold_1']
        self.prefix_length_upv = self.config_dict['prefix_length_1']

        self.bit_number_upv = self.config_dict['bit_number']
        self.sigma_upv = self.config_dict['sigma']
        self.default_top_k_upv = self.config_dict['default_top_k']
        self.generator_model_name_upv = self.config_dict['generator_model_name']
        self.detector_model_name_upv = self.config_dict['detector_model_name']
        self.detect_mode_upv = self.config_dict['detect_mode']

    def initialize_parameters(self) -> None:
        print("Initialize Twinu(TS+UPV)-specific parameters.")

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Twinu'


class TwinuUtils:
    """Utility class for Twinu(TS+UPV) algorithm, contains helper functions."""

    def __init__(self, config: TwinuConfig, *args, **kwargs) -> None:
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

        self.generator_model = self._get_generator_model(
            self.config.bit_number_upv, self.config.prefix_length_upv + 1
        ).to(self.config.device)
        self.detector_model = self._get_detector_model(self.config.bit_number_upv).to(self.config.device)
        self.cache = {}
        self.top_k = self.config.gen_kwargs.get('top_k', self.config.default_top_k_upv)
        self.num_beams = self.config.gen_kwargs.get('num_beams', None)

    def _seed_rng_ts(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.config.seeding_scheme

        # using the last token and hash_key to generate random seed
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, \
                f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[self.config.lo_ts].item()
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
            gamma = self.gamma_network(self.embed_matrix[input_ids[self.config.lo_ts].item()])
            delta = self.delta_network(self.embed_matrix[input_ids[self.config.lo_ts].item()])
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
        green_token_mask = [-1] * int(self.config.prefix_length_ts)

        # todo: range(self.config.prefix_length_ts, len(input_ids))
        for idx in range(-1 * self.config.lo_ts, len(input_ids)+ self.config.lo_ts):
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

    def _get_generator_model(self, input_dim: int, window_size: int) -> UPVGenerator:
        """Load the generator model from the specified file."""
        model = UPVGenerator(input_dim, window_size)
        model.load_state_dict(torch.load(self.config.generator_model_name_upv))
        return model

    def _get_detector_model(self, bit_number: int) -> UPVDetector:
        """Load the detector model from the specified file."""
        model = UPVDetector(bit_number)
        model.load_state_dict(torch.load(self.config.detector_model_name_upv))
        return model

    def _get_predictions_from_generator(self, input_x: torch.Tensor) -> bool:
        """Get predictions from the generator model."""
        with torch.no_grad():
            output = self.generator_model(input_x)
            output = (output > 0.5).bool().item()
        return output

    def int_to_bin_list(self, n: int, length=8) -> list[int]:
        """Convert an integer to a binary list of specified length."""
        bin_str = format(n, 'b')[:length].zfill(length)
        return [int(b) for b in bin_str]

    def _select_candidates(self, scores: torch.Tensor) -> torch.Tensor:
        """Select candidate tokens based on the scores."""
        if self.num_beams is not None:
            threshold_score = torch.topk(scores, self.num_beams, largest=True, sorted=False)[0][-1]
            return (scores >= (threshold_score - self.config.delta_upv)).nonzero(as_tuple=True)[0]
        else:
            return torch.topk(scores, self.top_k, largest=True, sorted=False).indices

    def get_greenlist_ids_upv(self, input_ids: torch.Tensor, scores: torch.Tensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        greenlist_ids = []
        candidate_tokens = self._select_candidates(scores)

        # Ensure input_ids is a list for concatenation
        input_ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

        for v in candidate_tokens:
            # Now safely concatenate lists
            pair = input_ids_list[-self.config.prefix_length_upv:] + [v.item()] if self.config.prefix_length_upv > 0 else [
                v.item()]
            merged_tuple = tuple(pair)

            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                bin_list = [self.int_to_bin_list(num, self.config.bit_number_upv) for num in pair]
                result = self._get_predictions_from_generator(
                    torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
                )
                self.cache[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids

    def _judge_green(self, input_ids: torch.Tensor, current_number: int) -> bool:
        """Judge if the current token is green based on previous tokens."""

        # Get the last 'prefix_length' items from input_ids
        last_nums = input_ids[-self.config.prefix_length_upv:] if self.config.prefix_length_upv > 0 else []
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [self.int_to_bin_list(num, self.config.bit_number_upv) for num in pair]

        # load & update cache
        if merged_tuple in self.cache:
            result = self.cache[merged_tuple]
        else:
            result = self._get_predictions_from_generator(
                torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
            )
            self.cache[merged_tuple] = result

        return result

    def green_token_mask_and_stats(self, input_ids: torch.Tensor) -> tuple[list[bool], int, float]:
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
            if self._judge_green(input_ids[:idx], curr_token.item()):
                mask_list.append(1)  # Mark this token as green
                green_token_count += 1  # Increment the green token counter
            else:
                mask_list.append(0)  # Mark this token as not green

        # Compute the number of tokens that were evaluated for green status
        num_tokens_scored = len(input_ids) - self.config.prefix_length_upv

        # Calculate the z-score for the number of green tokens
        z_score = self._compute_z_score_upv(green_token_count, num_tokens_scored)

        # Return the mask list, count of green tokens, and the z-score
        return mask_list, green_token_count, z_score

    def _compute_z_score_upv(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma_upv
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count) + self.config.sigma_upv * self.config.sigma_upv * T)
        z = numer / denom
        return z


class TwinuLogitsProcessor(LogitsProcessor):
    def __init__(self, config: TwinuConfig, utils: TwinuUtils, *args, **kwargs) -> None:

        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(
            self,
            logits: torch.Tensor,
            greenlist_token_ids: torch.Tensor
    ) -> torch.Tensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(
            self, input_ids: torch.Tensor,
            scores: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.shape[-1] < self.config.prefix_length_ts:
            return scores

        if "opt" not in self.config.generation_model_name.lower():
            llama_str = self.config.tokenizer_llama.batch_decode(input_ids[:, -5:], add_special_tokens=False)
            input_ids_opt = self.config.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
            input_ids_ts = torch.tensor(input_ids_opt).to(self.config.device)
        else:
            input_ids_ts = input_ids

        for batch_idx in range(input_ids.shape[0]):
            greenlist_ids_upv = self.utils.get_greenlist_ids_upv(input_ids[batch_idx], scores=scores[batch_idx])
            greenlist_ids_ts, gamma_ts, delta_ts, gamma_len = self.utils.get_green_list_ids_ts(
                input_ids_ts[batch_idx], 'process'
            )
            delta_ts = delta_ts.to(dtype=scores.dtype)

            greenlist_ids_ts_set = set(greenlist_ids_ts.tolist())
            greenlist_ids_upv_set = set(greenlist_ids_upv)
            common_set = greenlist_ids_ts_set & greenlist_ids_upv_set
            common_ids = torch.tensor(list(common_set), dtype=torch.long)

            green_tokens_mask = self._calc_greenlist_mask(
                logits=scores[batch_idx], greenlist_token_ids=common_ids
            )
            scores[batch_idx][green_tokens_mask] = (scores[batch_idx][green_tokens_mask]
                                                    + delta_ts
                                                    + self.config.delta_upv)


        return scores


class Twinu(BaseWatermark):
    def __init__(self, algorithm_config: TwinuConfig) -> None:
        utils = TwinuUtils(algorithm_config)
        logits_processor = TwinuLogitsProcessor(algorithm_config, utils)
        super().__init__(algorithm_config, utils, logits_processor)
        self.config = algorithm_config
        self.utils = utils
        self.logits_processor = logits_processor

    def _detect_watermark_network_mode(self, encoded_text: torch.Tensor) -> tuple[bool, None]:
        # Convert input IDs to binary sequence
        inputs_bin = [self.utils.int_to_bin_list(token_id, self.config.bit_number_upv) for token_id in encoded_text]
        inputs_bin = torch.tensor(inputs_bin, device=self.config.device)

        # Run the model on the input binary sequence
        outputs = self.utils.detector_model(inputs_bin.unsqueeze(dim=0).float())
        outputs = outputs.reshape([-1])
        predicted = (outputs.data > 0.5).int()

        # Determine watermark presence based on predictions
        is_watermarked = (predicted == 1).sum().item() > 0

        # z_score is not applicable in network mode
        return is_watermarked, None

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method
        z_score_ts, green_token_mask_ts = self.utils.score_sequence_ts(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked_ts: bool = z_score_ts > self.config.z_threshold_ts

        # Check the mode and perform detection accordingly
        if self.config.detect_mode_upv == 'key':
            _, _, z_score_upv = self.utils.green_token_mask_and_stats(encoded_text)
            # Determine if the z_score indicates a watermark
            is_watermarked_upv = z_score_upv > self.config.z_threshold_upv
        else:  # network
            is_watermarked_upv, z_score_upv = self._detect_watermark_network_mode(encoded_text)

        is_watermarked: bool = True if is_watermarked_upv or is_watermarked_ts else False
        # Return results based on the return_dict flag
        if return_dict:
            return {
                "is_watermarked1": is_watermarked_upv,
                "score1": z_score_upv,
                "is_watermarked2": is_watermarked_ts,
                "score2": z_score_ts,
            }
        else:
            return is_watermarked, (z_score_upv + z_score_ts) / 2
