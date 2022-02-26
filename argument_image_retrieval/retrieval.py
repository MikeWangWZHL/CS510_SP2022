import os
import json

from PIL import Image
from PIL import ImageFile

import numpy as np
import random
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import torch
import matplotlib.pyplot as plt
from glob import glob

DUMMY_IMAGE = Image.open('./dataset/images/I00/I00a534edc86bf5cd/image.png').convert('RGB')

def stance_reranking(model, processor, top_image_embeddings, step_one_ranking, stance):
    image_embeds = torch.from_numpy(top_image_embeddings).to(device)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    if stance == 'PRO':
        texts = ['Good; Positive; Agree']
    elif stance == 'CON':
        texts = ['Anti; Negative; Disagree']

    inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    text_embeds = outputs.text_embeds.detach()
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    # logits_per_image = logits_per_text.T
    logits_per_text = logits_per_text.detach().cpu().numpy()

    ranking = np.argsort(logits_per_text[0])[-10:]
    ranking = np.flip(ranking)
    step_two_ranking = [step_one_ranking[r] for r in ranking]
    return np.array(step_two_ranking)

def run_retrieval(model, processor, image_ids, image_embeddings, topic_prompts, topic_id_2_data, device, output_path = './output/test.txt', if_second_stage_reranking = False):
    # first stage ranking number
    if if_second_stage_reranking:
        k1 = 50
    else:
        k1 = 10

    image_embeds = torch.from_numpy(image_embeddings).to(device)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    result_lines = []
    for topic_id, topic in topic_prompts.items():
        if topic_id in ['1','2','3','4','8']:
            texts = [f"{topic['title']} {topic['PRO']}", f"{topic['title']} {topic['CON']}"]
            # print(topic_id,'\n',texts)
            inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds.detach()
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            # logits_per_image = logits_per_text.T
            logits_per_text = logits_per_text.detach().cpu().numpy()
            
            # first stage: top k1 Pro images
            ranking_pro = np.argsort(logits_per_text[0])[-k1:]
            ranking_pro = np.flip(ranking_pro)

            if if_second_stage_reranking:
                top_image_embeddings = image_embeddings[ranking_pro]
                # try second stage stance reranking, top 10
                ranking_pro = stance_reranking(model, processor, top_image_embeddings, ranking_pro, stance = 'PRO')

            for i in range(len(ranking_pro)):
                idx = ranking_pro[i]
                im_id = image_ids[idx]
                line = f'{topic_id} PRO I{im_id} {i} {logits_per_text[0][idx]} test-v1\n'
                result_lines.append(line)
        
            # first stage: top k1 Con images
            ranking_con = np.argsort(logits_per_text[1])[-k1:]
            ranking_con = np.flip(ranking_con)
            
            if if_second_stage_reranking:
                top_image_embeddings = image_embeddings[ranking_con]
                # try second stage stance reranking, top 10
                ranking_con = stance_reranking(model, processor, top_image_embeddings, ranking_con, stance = 'CON')

            for i in range(len(ranking_con)):
                idx = ranking_con[i]
                im_id = image_ids[idx]
                line = f'{topic_id} CON I{im_id} {i} {logits_per_text[1][idx]} test-v1\n'
                result_lines.append(line)

            with open(output_path, 'w') as out:
                for line in result_lines:
                    out.write(line)

            
        


if __name__ == '__main__':
    ''' load image embedding'''
    image_embeddings_npy_path = './embeddings/clip-vit-base-patch32/image_embeddings.npy'
    image_ids_json_path = './embeddings/clip-vit-base-patch32/image_ids_list.json'
    image_ids = json.load(open(image_ids_json_path))
    image_embeddings = np.load(image_embeddings_npy_path)
    print('embedding shape:', image_embeddings.shape)
    
    ''' load topic and prompts'''
    topic_prompts_json_path = './prompts/topic_prompts_v1.json'
    topic_id_2_data_path = './topic_id_2_topic.json'
    topic_prompts = json.load(open(topic_prompts_json_path))
    topic_id_2_data = json.load(open(topic_id_2_data_path))

    ''' output path '''
    output_path = './output/test-v1-with-stance-reranking-k1-30.txt'

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    ''' set up model'''
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ''' run retrieval '''
    run_retrieval(model, processor, image_ids, image_embeddings, topic_prompts, topic_id_2_data, device, output_path = output_path)