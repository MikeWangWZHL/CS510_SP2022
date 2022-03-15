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
import pickle

DUMMY_IMAGE = Image.open('./dataset/images/I00/I00a534edc86bf5cd/image.png').convert('RGB')

def stance_reranking_image(model, processor, top_image_embeddings, step_one_ranking, stance):
    image_embeds = torch.from_numpy(top_image_embeddings).to(device)
    # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
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

def run_retrieval_imageonly_stance_prompt(model, processor, image_ids, image_embeddings, topic_prompts, topic_id_2_data, device, output_path = './output/test.txt', if_second_stage_reranking = False):
    # first stage ranking number
    if if_second_stage_reranking:
        k1 = 50
    else:
        k1 = 10

    image_embeds = torch.from_numpy(image_embeddings).to(device)
    # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
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


# compute pairwise similarity between topic prompt and 
def compute_similarity(prompt_emb, arg_embedds, logit_scale):
    score = 0
    for emb in arg_embedds:
        score += np.dot(prompt_emb, emb) * logit_scale
    score /= len(arg_embedds)
    return score

def get_argument_text_similarity(prompt_text_embeds, argument_text_embeds, image_ids, logit_scale):

    '''
        prompt_text_embeds: (2, hidden_size)
        argument_text_embeds: {im_id:}
        return: matrix: (2, num_image_id)
    '''
    ret = np.zeros((2, len(image_ids)))
    for i in range(len(image_ids)):
        im_id = image_ids[i]
        arg_text_embeds = argument_text_embeds[im_id]
        if arg_text_embeds is not None: # if not None
            # pro
            ret[0][i] = compute_similarity(prompt_text_embeds[0],arg_text_embeds, logit_scale)
            # con
            ret[1][i] = compute_similarity(prompt_text_embeds[1],arg_text_embeds, logit_scale)
    return ret

def stance_reranking_argument_text(model, processor, argument_text_embeds, top_image_ids, step_one_ranking, stance):

    # texts = ['Good; Positive; Agree','Anti; Negative; Disagree']
    texts = ['good','bad']

    inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    prompt_text_embeds = outputs.text_embeds.detach().cpu().numpy()
    # print(texts)
    # print(prompt_text_embeds.shape) # (2, 512)
    # print(top_image_ids)
    matrix = get_argument_text_similarity(prompt_text_embeds, argument_text_embeds, top_image_ids, logit_scale=100)
    # print(matrix.shape)
    # print(matrix)
    for i in range(len(matrix[0])):
        if matrix[0][i] == 0 or matrix[1][i]==0:
            matrix[0][i] = 0.5
            matrix[1][i] = 0.5
    pro_scores = matrix[0,:] / (matrix[0,:]+matrix[1,:])
    if stance == 'PRO':
        scores = pro_scores
    elif stance == 'CON':
        scores = np.flip(pro_scores)
    ranking = np.argsort(scores)[-10:]
    ranking = np.flip(ranking)
    step_two_ranking = [step_one_ranking[r] for r in ranking]
    return np.array(step_two_ranking)

def run_retrieval_with_argumenttext_stance_prompt(model, processor, image_ids, image_embeddings, argument_text_embeds, alpha, topic_prompts, topic_id_2_data, device, output_path = './output/test.txt', if_second_stage_reranking = False):
    # first stage ranking number
    if if_second_stage_reranking:
        k1 = 50
    else:
        k1 = 10

    image_embeds = torch.from_numpy(image_embeddings).to(device)
    # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    result_lines = []
    for topic_id, topic in topic_prompts.items():
        if topic_id in ['1','2','3','4','8']:
            texts = [f"{topic['title']} Yes. {topic['PRO']}", f"{topic['title']} No. {topic['CON']}"]
            # print(topic_id,'\n',texts)
            inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            prompt_text_embeds = outputs.text_embeds.detach()

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_text = torch.matmul(prompt_text_embeds, image_embeds.t()) * logit_scale
            # logits_per_image = logits_per_text.T
            logits_per_text = logits_per_text.detach().cpu().numpy()

            # add textual similarity
            logit_scale = logit_scale.detach().cpu().numpy()
            prompt_text_embeds = prompt_text_embeds.cpu().numpy()

            logits_per_text_argument_text_similarity = get_argument_text_similarity(prompt_text_embeds, argument_text_embeds, image_ids, logit_scale)
            # print(logits_per_text)
            # print()
            # print(logits_per_text_argument_text_similarity)
            logits_per_text = alpha * logits_per_text + (1-alpha) * logits_per_text_argument_text_similarity
            # print()
            # print(logits_per_text)
            # first stage: top k1 Pro images
            ranking_pro = np.argsort(logits_per_text[0])[-k1:]
            ranking_pro = np.flip(ranking_pro)

            if if_second_stage_reranking:
                top_image_ids = [image_ids[r] for r in ranking_pro]
                # try second stage stance reranking, top 10
                ranking_pro = stance_reranking_argument_text(model, processor, argument_text_embeds, top_image_ids, ranking_pro, stance = 'PRO')

            for i in range(len(ranking_pro)):
                idx = ranking_pro[i]
                im_id = image_ids[idx]
                line = f'{topic_id} PRO I{im_id} {i} {logits_per_text[0][idx]} test\n'
                result_lines.append(line)
        
            # first stage: top k1 Con images
            ranking_con = np.argsort(logits_per_text[1])[-k1:]
            ranking_con = np.flip(ranking_con)
            
            if if_second_stage_reranking:
                top_image_ids = [image_ids[r] for r in ranking_con]
                # try second stage stance reranking, top 10
                ranking_con = stance_reranking_argument_text(model, processor, argument_text_embeds, top_image_ids, ranking_con, stance = 'CON')

            for i in range(len(ranking_con)):
                idx = ranking_con[i]
                im_id = image_ids[idx]
                line = f'{topic_id} CON I{im_id} {i} {logits_per_text[1][idx]} test\n'
                result_lines.append(line)

            with open(output_path, 'w') as out:
                for line in result_lines:
                    out.write(line)

def run_retrieval_with_argumenttext_stance_prompt_topic_ranking_first(model, processor, image_ids, image_embeddings, argument_text_embeds, alpha, topic_k, topic_prompts, topic_id_2_data, device, output_path = './output/test.txt'):
    # first stage ranking number
    print('topic k:', topic_k)

    image_embeds = torch.from_numpy(image_embeddings).to(device)
    # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    result_lines = []
    for topic_id, topic in tqdm(topic_prompts.items()):
        if topic_id in ['1','2','3','4','8']:
            texts = [f"{topic['title']} Yes!", f"{topic['title']} No!"]
            # print(topic_id,'\n',texts)
            inputs = processor(text=texts, images=DUMMY_IMAGE, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            prompt_text_embeds = outputs.text_embeds.detach()

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_text = torch.matmul(prompt_text_embeds, image_embeds.t()) * logit_scale
            # logits_per_image = logits_per_text.T
            logits_per_text = logits_per_text.detach().cpu().numpy()

            # add textual similarity
            logit_scale = logit_scale.detach().cpu().numpy()
            prompt_text_embeds = prompt_text_embeds.cpu().numpy()

            logits_per_text_argument_text_similarity = get_argument_text_similarity(prompt_text_embeds, argument_text_embeds, image_ids, logit_scale)
            logits_per_text = alpha * logits_per_text + (1-alpha) * logits_per_text_argument_text_similarity # (2,23841)
            
            topic_similarity = logits_per_text[0] + logits_per_text[1] # (23841)
            # select top topic related ids
            topic_ranking = np.argsort(topic_similarity)[-topic_k:]
            topic_ranking = np.flip(topic_ranking)

            selected_logits = np.zeros((2,len(topic_ranking)))
            selected_logits[0] = logits_per_text[0][topic_ranking]
            selected_logits[1] = logits_per_text[1][topic_ranking]

            pro_scores = selected_logits[0,:] / (selected_logits[0,:]+selected_logits[1,:])
            con_scores = selected_logits[1,:] / (selected_logits[0,:]+selected_logits[1,:])
            
            # get pro ranking
            pro_ranking_sub = np.argsort(pro_scores)[-10:]
            pro_ranking_sub = np.flip(pro_ranking_sub)
            pro_ranking = [topic_ranking[r] for r in pro_ranking_sub]
            for i in range(len(pro_ranking)):
                idx = pro_ranking[i]
                im_id = image_ids[idx]
                line = f'{topic_id} PRO I{im_id} {i} {logits_per_text[0][idx]} test\n'
                result_lines.append(line)
            
            # get con ranking
            con_ranking_sub = np.argsort(con_scores)[-10:]
            con_ranking_sub = np.flip(con_ranking_sub)
            con_ranking = [topic_ranking[r] for r in con_ranking_sub]
            for i in range(len(con_ranking)):
                idx = con_ranking[i]
                im_id = image_ids[idx]
                line = f'{topic_id} CON I{im_id} {i} {logits_per_text[0][idx]} test\n'
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

    ''' load argument text sentences '''
    # canery 
    image_id_2_argument_text_path = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/page_text_arguments/image_id_2_argument_sentences.json'
    image_id_2_argument_text = json.load(open(image_id_2_argument_text_path))
    image_id_2_argument_text_embedding_path = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/page_text_arguments/embeddings_top-5.pickle'
    with open(image_id_2_argument_text_embedding_path, 'rb') as handle:
        image_id_2_argument_text_embedding = pickle.load(handle)
    
    ''' output path '''
    output_path = './output/test_add_argument_text'

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
    # run_retrieval(model, processor, image_ids, image_embeddings, topic_prompts, topic_id_2_data, device, output_path = output_path)

    # alpha = 1
    # reranking = True
    # run_retrieval_with_argumenttext_stance_prompt(
    #     model, 
    #     processor, 
    #     image_ids, 
    #     image_embeddings, 
    #     image_id_2_argument_text_embedding, 
    #     alpha, 
    #     topic_prompts, 
    #     topic_id_2_data, 
    #     device, 
    #     output_path = f'./output/test_with_arg_alpha-{alpha}_reranking-{reranking}.txt', 
    #     if_second_stage_reranking = reranking
    # )

    alpha = 1
    topic_k = 40
    run_retrieval_with_argumenttext_stance_prompt_topic_ranking_first(
        model, 
        processor, 
        image_ids, 
        image_embeddings, 
        image_id_2_argument_text_embedding, 
        alpha, 
        topic_k,
        topic_prompts, 
        topic_id_2_data, 
        device,
        output_path = f'./output/test_with_arg_topic_ranking_first-{topic_k}_alpha-{alpha}.txt'
    )