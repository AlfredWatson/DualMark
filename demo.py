import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from watermark.auto_watermark import WATERMARK_MAPPING_NAMES
from utils.transformers_config import TransformersConfig


if __name__ == '__main__':

    watermark_name = 'Twins'

    text = "Please input some text to watermark model."
    
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
    watermarked_text_detect = watermark_model.detect_watermark(watermarked_text)
    print(watermarked_text_detect)
