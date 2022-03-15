import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import shutil
from tqdm import tqdm

def topic_id_2_topic(topic_xml_path):
    tree = ET.parse(topic_xml_path)
    root = tree.getroot()
    topicid2topic_dict = {}
    for topic in root:
        t = {}
        for item in topic:
            t[item.tag] = item.text.strip()
        topicid2topic_dict[t['number']] = t
    return topicid2topic_dict

def visualize_result(result_txt, image_root, output_vis_dir_root):
    topic_2_img_ids = {}
    with open(result_txt) as f:
        for line in f:
            parsed_line = line.split(' ')
            topic_id, stance, img_id = parsed_line[:3]
            if topic_id not in topic_2_img_ids:
                topic_2_img_ids[topic_id] = {
                    "PRO":[],
                    "CON":[]
                }
            else:
                topic_2_img_ids[topic_id][stance].append(img_id)
    print(topic_2_img_ids)
    print('copying images ...')
    for topic, result in tqdm(topic_2_img_ids.items()):
        # copy PRO images
        output_dir = os.path.join(output_vis_dir_root,f"{topic}-PRO")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for img_id in result['PRO']:
            src = os.path.join(image_root,f'{img_id[:3]}/{img_id}/image.png')
            dest = os.path.join(output_dir,f'{img_id}.png')
            shutil.copyfile(src, dest)
        # copy CON images
        output_dir = os.path.join(output_vis_dir_root,f"{topic}-CON")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for img_id in result['CON']:
            src = os.path.join(image_root,f'{img_id[:3]}/{img_id}/image.png')
            dest = os.path.join(output_dir,f'{img_id}.png')
            shutil.copyfile(src, dest)

def topic_prompts_empty_template(topicid2topic_dict):
    dict_ = {str(i+1):{"title":topicid2topic_dict[str(i+1)]['title'],"PRO":"","CON":""} for i in range(50)}
    
    with open('topic_prompts_empty.json','w') as out:
        json.dump(dict_,out,indent=4)


if __name__ == "__main__":
    ''' topic.xml to json '''
    # topicid2topic_dict = topic_id_2_topic('/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/dataset/topics.xml')
    # with open('topic_id_2_topic.json','w') as out:
    #     json.dump(topicid2topic_dict, out, indent=4)

    ''' visualize result '''
    # result_txt = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/output/given_baseline/run.txt'
    # result_txt = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/output/test-v1.txt'
    # result_txt = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/output/test-v1-with-stance-reranking-k1-50.txt'
    # result_txt = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/output/test-v1-with-stance-reranking-k1-30.txt'
    topic_k = 20
    for alpha in [0.1]:
        result_name = f'test_with_arg_topic_ranking_first-{topic_k}_alpha-{alpha}'
        result_txt = f'/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/output/{result_name}.txt'
        image_root = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/dataset/images'
        output_vis_dir_root = f'./visualization/{result_name}'
        visualize_result(result_txt, image_root, output_vis_dir_root)

    ''' topic prompts empty json '''
    # topic_prompts_empty_template(json.load(open('/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/topic_id_2_topic.json')))