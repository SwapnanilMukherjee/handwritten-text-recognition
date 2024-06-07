import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import jiwer

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import transformers
from datasets import load_metric
from transformers import TrOCRProcessor, TrOCRConfig
from transformers import VisionEncoderDecoderModel
from transformers import AdamW

import os
import xml.etree.ElementTree as ET 
import rasterio
from tqdm import tqdm
import pickle 


# disabling some warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter('ignore', UserWarning)


cvl_data_path = '../handwritten-text-recognition/cvl-database-1-1/'
cvl_cropped_images =  cvl_data_path+'cvl-database-cropped-1-1/'
root_dir = '../handwritten-text-recognition/'

# loading ghe csv files containing the (image) file names and the corresponding text label (string)
# trainset_line = pd.read_csv(cvl_data_path+'lines_trainset.csv')
# testset_line = pd.read_csv(cvl_data_path+'lines_testset.csv')

# trainset_page = pd.read_csv(cvl_data_path+'page_trainset.csv')
# testset_page = pd.read_csv(cvl_data_path+'page_testset.csv')
# # some XML files are faulty and do not contain label information. XML parses returns NaN. These NaN values are replaced by the following text. Can be handled in a better way
# testset_page.fillna('Text not available', inplace=True)

PAGE = False # change this parameter as required; if training on page-data this should be True, otherwise false. 

class CVLDataset(Dataset):
    def __init__(self, processor, split=None, page=False):
        self.processor = processor
        
        if page:
            self.img_dir = cvl_cropped_images
            self.max_target_length = 100
            if split.lower() == 'train':
                self.df = pd.read_csv(root_dir+'page_trainset.csv')
            elif split.lower() == 'test':
                self.df = pd.read_csv(root_dir+'page_testset.csv')
                # some XML files are faulty and do not contain label information. XML parses returns NaN. These NaN values are replaced by the following text
                self.df.fillna('Text not available', inplace=True)
        else:
            self.img_dir = cvl_data_path+split+'set/lines/'
            self.max_target_length = 25
            if split.lower() == 'train':
                self.df = pd.read_csv(root_dir+'lines_trainset.csv')
            elif split.lower() == 'test':
                self.df = pd.read_csv(root_dir+'lines_testset.csv')
                
        self.page = page

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, return_orig=False):
        file_name, text = self.df.iloc[idx, :]
        if self.page == False:
            folder = file_name.split("-")[0]
            image = rasterio.open(self.img_dir+folder+'/'+file_name)
            image = image.read()
        else:
            image = rasterio.open(self.img_dir+file_name)
            image = image.read()
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length,
                                          truncation=True,
                                         ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        
        if return_orig:
            return image, text, encoding
        else:
            return encoding


# loading the model's data processor 
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

train_data = CVLDataset(split='train', page=PAGE, processor=processor)
test_data = CVLDataset(split='test', page=PAGE, processor=processor)
print("Number of training examples:", len(train_data))
print("Number of testing examples:", len(test_data))

train_loader = DataLoader(train_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=18)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, pin_memory=True, num_workers=18)


# visualizing a random sample; sanity check 
image, text, encoding = train_data.__getitem__(111, return_orig=True)
plt.imshow(image.transpose(1,2,0))
print(text)

print(encoding['pixel_values'].shape, encoding['labels'].shape)


cer_metric = load_metric("cer")
# computes the cer and wer score between recognized text from input image and actual label [batched]
def compute_metrics(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = jiwer.wer(label_str, pred_str)
    
    return cer, wer

# initializing gradient scaler for half-precision training
scaler = torch.cuda.amp.GradScaler()

# trains the model for one epoch
def train(model, dataloader, optimizer, scheduler=None):
    model.train()
    
    running_loss = 0.0
    nb_tr_steps = 0
    
    for batch in tqdm(dataloader):
        # move pixel values to the GPU
        for k,v in batch.items():
            batch[k] = v.to(device, non_blocking=True)

        with torch.autocast(dtype=torch.float16, device_type=device):
            outputs = model(**batch)
            loss = outputs.loss 
        
#         loss.backward()
        scaler.scale(loss).backward()
#         optimizer.step()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()
        
        if scheduler != None:
            scheduler.step()
        
        running_loss += loss
        nb_tr_steps += 1
    
    return model, running_loss/nb_tr_steps


def evaluate(model, dataloader):
    model.eval()
    
    running_loss = 0.0
    nb_tr_steps = 0.0
    tmp_eval_cer, tmp_eval_wer = 0.0, 0.0
    output_ids = []
    
    for batch in tqdm(dataloader):
        # move pixel values to the GPU
        for k,v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        
        with torch.autocast(dtype=torch.float16, device_type=device):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
        
        b_output_ids = model.generate(batch["pixel_values"].to(device))
        output_ids.extend(b_output_ids)
        cer, wer = compute_metrics(pred_ids=b_output_ids, label_ids=batch["labels"])
        
        running_loss += loss
        tmp_eval_cer += cer
        tmp_eval_wer += wer
        nb_tr_steps += 1
    
    return running_loss/nb_tr_steps, tmp_eval_cer/nb_tr_steps, tmp_eval_wer/nb_tr_steps, output_ids

# selects device; but because half precision trainig is being used, the code may break without a GPU
if torch.cuda.is_available():
    device = 'cuda' 
else:
    device = 'cpu'


def get_model(path, train_page=False):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    model.to(device)
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    if train_page:
        max_length = 100
    else:
        max_length = 25
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 5
    
    return model 


def get_optimizer(model, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    
    return optimizer

def run(model_path, epochs):

    model = get_model(model_path, train_page=PAGE)
    optimizer = get_optimizer(model, lr=1e-6)

    tr_loss_set, val_loss_set = [], []
    cer_set, wer_set = [], []
    
    best_cer, best_wer = 100,100

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs), end='\t')
        model, tr_loss = train(model, test_loader, optimizer)
        val_loss, cer, wer, str_outputs = evaluate(model, train_loader)

        tr_loss_set.append(tr_loss.item())
        val_loss_set.append(val_loss.item())
        cer_set.append(cer)
        wer_set.append(wer)

        print("Train loss:{}".format(tr_loss), end='\t')
        print("Val loss:{}".format(val_loss), end='\t')
        print("CER:{}".format(cer), end='\t')
        print("WER:{}".format(cer))
        print("==============================================================================")
        
        if cer < best_cer and wer < best_wer:
            print("Better model state found. Checkpointing...")
            if not PAGE:
                torch.save(model.state_dict(), root_dir+'checkpoints/'+model_path.split('/')[1]+'_finetuned_line-level.pth')
            else:
                torch.save(model.state_dict(), root_dir+'checkpoints/'+model_path.split('/')[1]+'_finetuned_page-level.pth')
            best_cer = cer
            best_wer = wer 
            best_outputs = str_outputs
        if cer < best_wer:
            best_cer = cer
        if wer < best_wer:
            best_wer = wer
    
    output_str = processor.tokenizer.batch_decode(best_outputs, skip_special_tokens=True)
    print(output_str)

    with open(root_dir+"recognized_text.pickle", 'wb') as f:
        pickle.dump(output_str, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tr_loss_set, val_loss_set, cer_set, wer_set
    
epochs = 10
tr_losses, val_losses, cer_scores, wer_scores = [],[],[],[]

model_path = 'microsoft/trocr-base-handwritten'
tr_loss_set, val_loss_set, cer_set, wer_set = run(model_path, epochs)

tr_losses.append(tr_loss_set)
val_losses.append(val_loss_set)
cer_scores.append(cer_set)
wer_scores.append(wer_set)


