# DualMark
A Novel Dual Watermarking Approach for Large Language Models

## Overview

Watermarking is a promising approach to detect AI-generated text by embedding imperceptible, machine-detectable patterns. However, existing green–red watermarking methods often fail under adversarial edits such as paraphrasing or context manipulation.

**DualMark** is a novel **dual-watermarking framework** designed to overcome these limitations by embedding **two independent green–red watermarks** using separate context windows. During generation, only tokens appearing in **both green lists** are biased, while detection succeeds if **either watermark** yields a sufficient green-token signal. This overlapping design makes the watermark **more robust** and **harder to remove**, without sacrificing text quality.

## Key Features

✅ **Dual Detection Paths**: Detection succeeds if either of two embedded watermarks is preserved.

✅ **Stronger Resistance to Edits**: Enhanced robustness against word-level and sentence-level attacks (e.g., substitution, deletion, paraphrasing).

✅ **Low Overhead**: Easily integrates with existing watermarking frameworks with minimal computation cost.

✅ **Flexible Composition**: Can be used with two different methods (e.g., KGW + UPV), or two instances of the same method (e.g., KGW + KGW).

✅ **Preserves Text Quality**: Maintains similar levels of fluency, coherence, and semantic integrity as baseline methods.


# Requirements and Installation

To run the code, you will need to set up a Python environment with the required dependencies. You can use `pip` to install the necessary libraries (已经在`requirements.txt`中给出).

## Base Code

Our code fits perfectly into the watermarking framework [MarkLLM][MarkLLM-code], you can copy the code of this project directly into its directory and run tests.

## Model Download

The baseline model we use in the paper is available at the following link:[TSW][TSW-code], [UPV][UPV-code].

# Usage

## Embedding Watermark

To get the watermark text, use the following command:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from watermark.auto_watermark import WATERMARK_MAPPING_NAMES
from utils.transformers_config import TransformersConfig


watermark_name = 'Twins'

text = "Please input some prompt to watermark model."

assert watermark_name in WATERMARK_MAPPING_NAMES, f"{watermark_name} not in WATERMARK_MAPPING_NAMES"
# Device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
generate_model_path = "facebook/opt-1.3b"
# Transformers config
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(generate_model_path).to(device),
    tokenizer=AutoTokenizer.from_pretrained(generate_model_path),
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4
)
# Load watermark algorithm
watermark_model = AutoWatermark.load(
    f'{watermark_name}',
    transformers_config=transformers_config,
    algorithm_config=f'./config/{watermark_name}.json',
)

watermarked_text = watermark_model.generate_watermarked_text(text)
print(watermarked_text)
```

## Detecting Watermark

To detect the watermark in a given text, use the following command:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from watermark.auto_watermark import WATERMARK_MAPPING_NAMES
from utils.transformers_config import TransformersConfig


watermark_name = 'Twins'

text = "Please input some text for detection."

assert watermark_name in WATERMARK_MAPPING_NAMES, f"{watermark_name} not in WATERMARK_MAPPING_NAMES"
# Device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
generate_model_path = "facebook/opt-1.3b"
# Transformers config
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(generate_model_path).to(device),
    tokenizer=AutoTokenizer.from_pretrained(generate_model_path),
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4
)
# Load watermark algorithm
watermark_model = AutoWatermark.load(
    f'{watermark_name}',
    transformers_config=transformers_config,
    algorithm_config=f'./config/{watermark_name}.json',
)

watermarked_text_detect = watermark_model.detect_watermark(text)
print(watermarked_text_detect)
```

# Experimental Results

The results of our original experiments are given in the folder `result`. After statistical analysis, **DualMark** shows significant robustness improvement in several attack scenarios:

- Word-level attack: Under word replacement/deletion attacks ranging from 10% to 50%, the DualMark combination (e.g., KGW+TSW) consistently outperforms a single method in AUROC, TPR@5%FPR, and F1-score metrics.

- Document-level attack: In the face of sentence-level rewriting by Dipper attack, DualMark still maintains excellent detection results when the paraphrasing ratio is as high as 100%.

- Text quality: DualMark is comparable to the best existing methods in terms of Perplexity, Semantic Loss, Coherence, etc., proving that it does not significantly degrade text quality.

For detailed experimental results and graphs, please see the main paper.

# Citation

If you use this code in your research, please cite the following paper:

```bib
@inproceedings{XXX,
  title={DualMark: A Novel Dual Watermarking Approach for Large Language Models},
  author={Jipeng Qiang, Jifei Hao, Yun Li, Yi Zhu, Yunhao Yuan, Xiaoye Ouyang},
  year={202X},
}
```

---
If you have any question about how to run the code. Please contact haojifei2000@gmail.com


[MarkLLM-code]: https://github.com/THU-BPM/MarkLLM
[TSW-code]: https://github.com/mignonjia/TS_watermark
[UPV-code]: https://github.com/THU-BPM/unforgeable_watermark