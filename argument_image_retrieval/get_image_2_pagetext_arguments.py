import os
import json
import numpy as np
import random
from tqdm import tqdm
from glob import glob
from canary.argument_pipeline import download_model, load_model, analyse_file
import spacy

from PIL import Image
from PIL import ImageFile

from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import torch
import matplotlib.pyplot as plt

import pickle

def load_page_texts(image_dir_root):
    subdirs = glob(os.path.join(image_dir_root,"*"))
    subsubdirs = []
    for subdir in subdirs:
        subsubdirs += glob(os.path.join(subdir,"*"))
    image_ids = []
    page_texts_list = [] # list of list
    subsubdirs = sorted(subsubdirs)

    for subsubdir in tqdm(subsubdirs):
        image_ids.append(os.path.basename(subsubdir)[1:])
        # images.append(Image.open(os.path.join(subsubdir,'image.png')).convert('RGB')) # PIL image
        pages = glob(os.path.join(subsubdir,'pages/*'))
        page_texts = []
        for page in pages:
            file_path = os.path.join(page,'snapshot/text.txt')
            page_texts.append(file_path)
        page_texts_list.append(page_texts)

    assert len(image_ids) == len(page_texts_list)
    print('num of images:', len(page_texts_list))
    return image_ids, page_texts_list

def get_image_2_argument_sentence(image_ids, page_texts_list, threshold):
    detector = load_model("argument_detector")
    nlp = spacy.load("en_core_web_sm", disable=['ner','tagger','lemmatizer'])
    ret = {}
    for i in tqdm(range(len(image_ids))):
        im_id = image_ids[i]
        file_paths = page_texts_list[i]
        sentences = []
        for fp in file_paths:
            with open(fp, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        doc = nlp(line)
                        for sent in doc.sents:
                            if len(sent.text) > 10:
                                try:
                                    pred = detector.predict(sent.text, probability=True)
                                    if pred[True] >= threshold:
                                        sentences.append((sent.text,pred[True]))
                                except:
                                    print('error! contunie...')
                                    continue
        ret[im_id] = sentences
    return ret


def main():
    ''' image root '''
    image_root_path = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/dataset/images'
    ''' output dir '''
    output_dir = "./page_text_arguments"
    ''' threshold '''
    threshold = 0.95

    ''' set up device '''
    # # use cuda
    # if torch.cuda.is_available():  
    #     dev = "cuda:3" 
    # else:  
    #     dev = "cpu"
    # device = torch.device(dev)

    print('loading files...')
    image_ids, page_texts_list = load_page_texts(image_root_path)
    print('detecting...')
    image_id_2_argument_sentences = get_image_2_argument_sentence(image_ids, page_texts_list, threshold)

    # save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir,'image_id_2_argument_sentences.json'),'w') as out:
        json.dump(image_id_2_argument_sentences,out,indent=4)

def get_argument_text_embeddings():
    DUMMY_IMAGE = Image.open('./dataset/images/I00/I00a534edc86bf5cd/image.png').convert('RGB')

    image_id_2_argument_sentences = json.load(open("./page_text_arguments/image_id_2_argument_sentences.json"))
    output_dir = "./page_text_arguments"
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    ''' set up model '''
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    ''' get embedding for top k'''
    topk = 5
    ret = {}
    for key, sents in tqdm(image_id_2_argument_sentences.items()):
        if sents == []:
            ret[key] = None
            continue
        # print(key)
        sents = sorted(sents, key=lambda x:x[1], reverse=True)
        sents = [item[0] for item in sents[:topk]]
        # print(sents)
        inputs = processor(
            text=sents, images=DUMMY_IMAGE, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model(**inputs)
        # print(outputs.text_embeds.size())
        ret[key] = outputs.text_embeds.detach().cpu().numpy()
        # print(ret)
        # break
    
    # save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir,f'embeddings_top-{topk}.pickle'), 'wb') as handle:
        pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    get_argument_text_embeddings()
