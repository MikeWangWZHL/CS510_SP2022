import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import json

from PIL import Image
from PIL import ImageFile

import numpy as np
import random
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
from glob import glob
import pickle

from collections import defaultdict

class StanceDataset(Dataset):
    def __init__(self, traning_samples_txt, topic_prompt_json, image_id_2_embedding):
        self.topic_prompt_json = topic_prompt_json
        self.annotation = self._load_training_samples_ann(traning_samples_txt)
        self.image_id_2_embedding = image_id_2_embedding

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        image_id, text, label = self.annotation[idx]
        image_embedding = torch.from_numpy(self.image_id_2_embedding[image_id])
        label = torch.tensor(label).type(torch.LongTensor)
        
        return image_embedding, text, label

    def _load_training_samples_ann(self, traning_samples_txt):
        ann_dict = defaultdict(dict)
        with open(traning_samples_txt, 'r') as f:
            for line in f:
                parsed_line = line.split(' ')
                image_id = parsed_line[2][1:]
                topic_id = parsed_line[0]
                stance = parsed_line[1]
                label = parsed_line[3].strip()
                ann_dict[image_id][topic_id] = {'CON':False,'PRO':False}
                if label == '1':
                    ann_dict[image_id][topic_id][stance] = True
        ann = []
        for image_id,annotations in ann_dict.items():
            for topic_id, labels in annotations.items():
                CON_text = self.topic_prompt_json[topic_id]['CON']
                if CON_text == '':
                    CON_text = self.topic_prompt_json[topic_id]['title'] + ' ' + 'NO!'
                PRO_text = self.topic_prompt_json[topic_id]['PRO']
                if PRO_text == '':
                    PRO_text = self.topic_prompt_json[topic_id]['title'] + ' ' + 'YES!'

                if not labels['CON'] and not labels['PRO']:
                    ann.append((image_id, CON_text, 1))
                    ann.append((image_id, PRO_text, 1))
                elif not labels['CON'] and labels['PRO']:
                    ann.append((image_id, CON_text, 0))
                    ann.append((image_id, PRO_text, 2))
                elif labels['CON'] and not labels['PRO']:
                    ann.append((image_id, CON_text, 2))
                    ann.append((image_id, PRO_text, 0))
                else:
                    ann.append((image_id, CON_text, 2))
                    ann.append((image_id, PRO_text, 2))
        return ann

class StanceClassifier(nn.Module):
    def __init__(self, image_embed_size, text_embed_size):
        super(StanceClassifier, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(image_embed_size + text_embed_size, 3),
            nn.ReLU()
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(model, dataloader, loss_fn, optimizer, epoch, device, clip_model, processor):
    model.train()
    model.to(device)
    for e in range(epoch):
        for i, batch in enumerate(dataloader):
            image_embeds = batch[0].to(device)
            labels = batch[2].to(device)
            texts = batch[1]
            
            text_inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
            text_outputs = clip_model(**text_inputs)
            text_embeds = text_outputs.text_embeds
            # print(image_embeds.size())
            # print(text_embeds.size())
            # print(labels)
            image_text_embeds = torch.cat((image_embeds, text_embeds), dim=1)
            # print(image_text_embeds.size())
            pred = model(image_text_embeds)
            # print(pred.size())
            # print(pred)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                loss, current = loss.item(), i
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader):>5d}]")
    torch.save(model.state_dict(), "./checkpoint/model.pth")


DUMMY_IMAGE = Image.open('./dataset/images/I00/I00a534edc86bf5cd/image.png').convert('RGB')
def main():

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    ''' load image embedding'''
    image_embeddings_npy_path = './embeddings/clip-vit-base-patch32/image_embeddings.npy'
    image_ids_json_path = './embeddings/clip-vit-base-patch32/image_ids_list.json'
    image_ids = json.load(open(image_ids_json_path))
    image_embeddings = np.load(image_embeddings_npy_path)
    print('embedding shape:', image_embeddings.shape)
    image_id_2_embedding = {image_ids[i]:image_embeddings[i] for i in range(len(image_ids))}

    ''' load topic and prompts'''
    topic_prompts_json_path = './prompts/topic_prompts_empty.json'
    # topic_id_2_data_path = './topic_id_2_topic.json'
    topic_prompts = json.load(open(topic_prompts_json_path))
    # topic_id_2_data = json.load(open(topic_id_2_data_path))

    ''' dataset '''
    traning_samples_txt = './training-qrels.txt'
    dataset = StanceDataset(traning_samples_txt, topic_prompts, image_id_2_embedding)
    dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

    ''' clip model '''
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.eval()
    clip_model.to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    ''' model '''
    model = StanceClassifier(512,512) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    ''' train ''' 
    epoch = 20
    train(model, dataloader, loss_fn, optimizer, epoch, device, clip_model, processor)

def predict(image_id, topic_id, stance, checkpoint):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    ''' load image embedding'''
    image_embeddings_npy_path = './embeddings/clip-vit-base-patch32/image_embeddings.npy'
    image_ids_json_path = './embeddings/clip-vit-base-patch32/image_ids_list.json'
    image_ids = json.load(open(image_ids_json_path))
    image_embeddings = np.load(image_embeddings_npy_path)
    print('embedding shape:', image_embeddings.shape)
    image_id_2_embedding = {image_ids[i]:image_embeddings[i] for i in range(len(image_ids))}

    ''' load topic and prompts'''
    topic_prompts_json_path = './prompts/topic_prompts_empty.json'
    # topic_id_2_data_path = './topic_id_2_topic.json'
    topic_prompts = json.load(open(topic_prompts_json_path))
    # topic_id_2_data = json.load(open(topic_id_2_data_path))

    ''' clip model '''
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.eval()
    clip_model.to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    ''' model '''
    model = StanceClassifier(512,512) 
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    ''' input '''
    print('image id:', image_id)
    image_embed = torch.from_numpy(image_id_2_embedding[image_id]).to(device)
    
    CON_text = topic_prompts[topic_id]['CON']
    if CON_text == '':
        CON_text = topic_prompts[topic_id]['title'] + ' ' + 'NO!'
    PRO_text = topic_prompts[topic_id]['PRO']
    if PRO_text == '':
        PRO_text = topic_prompts[topic_id]['title'] + ' ' + 'YES!'
    if stance == 'CON':
        text = CON_text
    else:
        text = PRO_text
    print('text:', text)

    text_inputs = processor(text=[text], images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
    text_outputs = clip_model(**text_inputs)
    text_embed = text_outputs.text_embeds[0]

    image_text_embeds = torch.cat((image_embed, text_embed))

    pred = model(image_text_embeds).detach().cpu().numpy()
    print('pred:', pred)


if __name__ == "__main__":
    ''' main '''
    # main()

    ''' predict '''
    image_id = '31e4c262325e9018'
    topic_id = '1'
    stance = 'CON'
    checkpoint = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/checkpoint/model_3-29.pth'
    predict(image_id, topic_id, stance, checkpoint)