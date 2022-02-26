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

def load_images(image_dir_root):
    subdirs = glob(os.path.join(image_dir_root,"*"))
    subsubdirs = []
    for subdir in subdirs:
        subsubdirs += glob(os.path.join(subdir,"*"))
    image_ids = []
    images = []
    subsubdirs = sorted(subsubdirs)
    # n = 5
    for subsubdir in tqdm(subsubdirs):
        image_ids.append(os.path.basename(subsubdir)[1:])
        images.append(Image.open(os.path.join(subsubdir,'image.png')).convert('RGB')) # PIL image
        # if n == 1:
        #     break
        # n -= 1
    assert len(image_ids) == len(images)
    print('num of images:', len(images))
    return image_ids, images

def get_CLIP_embedding(image_dir_root, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_ids, images = load_images(image_dir_root)
    image_embeddings = []
    for image in tqdm(images):
        inputs = processor(
                text="hello world", images=image, return_tensors="pt", padding=True
            ).to(device)
        outputs = model(**inputs)
        image_embeddings.append(outputs.image_embeds[0].detach().cpu().numpy()) # (1,512)
    image_embeddings = np.array(image_embeddings)
    return image_ids, image_embeddings

if __name__ == '__main__':
    # handle ValueError: Decompressed Data Too Large
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ''' image root '''
    image_root_path = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/dataset/images'
    ''' output dir '''
    output_dir = "./embeddings/clip-vit-base-patch32"
    
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    image_ids, image_embeddings = get_CLIP_embedding(image_root_path,device)
    print(image_embeddings.shape)
    
    # save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir,'image_embeddings.npy'), image_embeddings)
    with open(os.path.join(output_dir,'image_ids_list.json'),'w') as out:
        json.dump(image_ids,out,indent=4)
