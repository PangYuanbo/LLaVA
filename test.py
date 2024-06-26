import locale
locale.getpreferredencoding = lambda: "UTF-8"
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

model_path = "liuhaotian/llava-v1.6-mistral-7b"

# 配置在CPU上运行，并避免使用FP4量化
kwargs = {"device_map": {"": "cpu"}}
kwargs['load_in_4bit'] = False  # 禁用4bit量化
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=False
)

# 加载模型
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
model = model.to(torch.device("cpu"))  # 确保模型在CPU上运行

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cpu')  # 确保视觉塔在CPU上运行
image_processor = vision_tower.image_processor

import requests
from PIL import Image
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

disable_torch_init()
conv_mode = "llava_v0"
conv = conv_templates[conv_mode].copy()
roles = conv.roles
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

def generate(img_url, inp):
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image = load_image(img_url)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].float()

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cpu"))
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(torch.device("cpu")),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs

    return ("\n", {"prompt": prompt, "outputs": outputs}, "\n")


