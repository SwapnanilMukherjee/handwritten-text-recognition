import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


import torch

import transformers
from transformers import TrOCRProcessor, TrOCRConfig
from transformers import VisionEncoderDecoderModel

import os
import rasterio
from tqdm import tqdm
from PIL import Image
import pickle 

image_folder =  '../handwritten-text-recognition/test_images/'
root_dir = '../handwritten-text-recognition/'

images = []
for file in sorted(os.listdir(image_folder)):
    img = Image.open(image_folder+file)
    images.append(img)
    
# images.pop(0) # the original full image is not required

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
pixel_values = processor(images, return_tensors="pt").pixel_values
print(pixel_values.shape)

inputs = {'pixel_values':pixel_values}

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
    
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
# model.load_state_dict(torch.load(root_dir+'checkpoints/trocr-base-handwritten_finetuned_line-level.pth'))
model.to(device)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 25
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 5

for k,v in inputs.items():
        inputs[k] = v.to(device, non_blocking=True)
output = model.generate(inputs['pixel_values'])

print(output.shape)

output_str = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
print(output_str)

with open(root_dir+"recognized_texts.pickle", 'wb') as file:
    pickle.dump(output_str, file, protocol=pickle.HIGHEST_PROTOCOL)

import spellcheck_ner
import build_schema