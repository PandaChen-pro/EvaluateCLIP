import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import os 
import time

os.environ['http_proxy'] = 'http://172.17.0.2:7890'
os.environ['https_proxy'] = 'http://172.17.0.2:7890'


FEATURES_DIR = "./saved_features"
os.makedirs(FEATURES_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 三个模型 ViT-L/14@336px, ViT-L/14, ViT-B/32
model_name = 'ViT-L/14' 
model, preprocess = clip.load(model_name, device=device)
model.eval()
print(f'Model {model_name} loaded on {device}')

def save_features(image_features, text_features, processed_image_ids, all_texts, text_to_image_mapping, image_id_to_text_indices, image_id_to_feature_idx):
    """保存所有特征和映射关系"""
    features_dict = {
        'image_features': image_features,
        'text_features': text_features,
        'processed_image_ids': processed_image_ids,
        'all_texts': all_texts,
        'text_to_image_mapping': text_to_image_mapping,
        'image_id_to_text_indices': image_id_to_text_indices,
        'image_id_to_feature_idx': image_id_to_feature_idx
    }
    torch.save(features_dict, os.path.join(FEATURES_DIR, f'flickr30k_{model_name.replace("/", "_")}_features.pt'))
    print("特征已保存到", os.path.join(FEATURES_DIR, f'flickr30k_{model_name.replace("/", "_")}_features.pt'))

def load_features():
    """加载保存的特征和映射关系"""
    features_path = os.path.join(FEATURES_DIR, f'flickr30k_{model_name.replace("/", "_")}_features.pt')
    if os.path.exists(features_path):
        print("正在加载已保存的特征...")
        features_dict = torch.load(features_path)
        return features_dict
    return None

features_dict = load_features()

if features_dict is None:
    print("未找到保存的特征，开始提取特征...")
    print("Loading Flickr30k dataset from Hugging Face...")
    try:
        hf_dataset = load_dataset("royokong/flickr30k_test", split="test", trust_remote_code=True, cache_dir="./hf_cache")
        print("Dataset loaded successfully.")
        print(f"Test set size: {len(hf_dataset)}")
        print("Dataset features:", hf_dataset.features)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you are logged in to Hugging Face Hub if required ('huggingface-cli login')")
        print("Or check the dataset name and availability.")
        exit()

    image_features_list = []
    text_features_list = []

    processed_image_ids = []
    all_texts = []
    text_to_image_mapping = []
    image_id_to_feature_idx = {} 
    image_id_to_text_indices = {} 

    print("Extracting image features...")
    current_image_idx = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(hf_dataset, desc="Processing Images")):
            image = item['image']
            image_id = item.get('image_id', str(i))
            if image.mode != "RGB":
                image = image.convert("RGB")
            try:
                image_input = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_list.append(image_features.cpu())
                processed_image_ids.append(image_id)
                image_id_to_feature_idx[image_id] = current_image_idx
                current_image_idx += 1
            except Exception as e:
                print(f"Error processing image {image_id} (index {i}): {e}")
                continue

    print("Preparing text data and extracting text features...")
    current_text_idx = 0
    text_batch_size = 64
    current_batch_texts = []

    for i, item in enumerate(tqdm(hf_dataset, desc="Preparing Texts & Encoding")):
        image_id = item.get('image_id', str(i))

        if image_id not in image_id_to_feature_idx:
            continue 

        captions = item['text']
        image_feature_idx = image_id_to_feature_idx[image_id]

        if image_id not in image_id_to_text_indices:
            image_id_to_text_indices[image_id] = []

        for caption in captions:
            prompted_caption = f"a photo of {caption}" 
            all_texts.append(prompted_caption)
            text_to_image_mapping.append(image_feature_idx) 
            image_id_to_text_indices[image_id].append(current_text_idx) 
            current_batch_texts.append(prompted_caption)
            current_text_idx += 1

            if len(current_batch_texts) >= text_batch_size:
                 with torch.no_grad():
                    text_inputs = clip.tokenize(current_batch_texts, truncate=True).to(device)
                    text_features = model.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features_list.append(text_features.cpu()) 
                 current_batch_texts = [] 

    if current_batch_texts:
         with torch.no_grad():
            text_inputs = clip.tokenize(current_batch_texts, truncate=True).to(device)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features.cpu())

    if not image_features_list:
         print("Error: No image features extracted.")
         exit()
    if not text_features_list:
         print("Error: No text features extracted.")
         exit()

    image_features_all = torch.cat(image_features_list, dim=0)
    text_features_all = torch.cat(text_features_list, dim=0)

    print(f"Total unique images processed: {image_features_all.shape[0]}")
    print(f"Total texts processed: {text_features_all.shape[0]}")
    print(f"Image features shape: {image_features_all.shape}")
    print(f"Text features shape: {text_features_all.shape}")

    save_features(image_features_all, text_features_all, processed_image_ids, all_texts,
                 text_to_image_mapping, image_id_to_text_indices, image_id_to_feature_idx)

else:
    print("Using loaded features...")
    image_features_all = features_dict['image_features']
    text_features_all = features_dict['text_features']
    processed_image_ids = features_dict['processed_image_ids']
    all_texts = features_dict['all_texts']
    text_to_image_mapping = features_dict['text_to_image_mapping']
    image_id_to_text_indices = features_dict['image_id_to_text_indices']
    image_id_to_feature_idx = features_dict['image_id_to_feature_idx']

    print(f"Loaded Image features shape: {image_features_all.shape}")
    print(f"Loaded Text features shape: {text_features_all.shape}")



num_texts = text_features_all.shape[0]
num_images = image_features_all.shape[0]
embedding_dim = text_features_all.shape[1]

similarity_matrix = torch.zeros(num_texts, num_images, dtype=image_features_all.dtype) # Use the same dtype

print(f"Initialized similarity matrix shape: {similarity_matrix.shape} on CPU")


text_batch_size_sim = 512 

print("Calculating similarity matrix in batches...")

image_features_gpu = image_features_all.to(device)

with torch.no_grad():
    for i in tqdm(range(0, num_texts, text_batch_size_sim), desc="Calculating Similarity"):
        text_batch_start = i
        text_batch_end = min(i + text_batch_size_sim, num_texts)
        
        text_features_batch = text_features_all[text_batch_start:text_batch_end, :].to(device)
        
  
        batch_similarity = (text_features_batch @ image_features_gpu.T)
        
        similarity_matrix[text_batch_start:text_batch_end, :] = batch_similarity.cpu()

print("Similarity matrix calculation complete.")
print(f"Final similarity matrix shape: {similarity_matrix.shape} on CPU")

def calculate_recall_hf(similarity_matrix, K_values, text_to_image_mapping, image_id_to_text_indices, processed_image_ids, image_id_to_feature_idx):
    """
    计算 Recall@K 指标
    Args:
        similarity_matrix (torch.Tensor): 文本-图像相似度矩阵 (num_texts, num_images) - Assumed on CPU
        K_values (list): 需要计算的 K 值列表, e.g., [1, 5, 10]
        text_to_image_mapping (list): 列表，索引是文本索引，值是对应的图像特征索引
        image_id_to_text_indices (dict): 映射 image_id -> list of text indices
        processed_image_ids (list): 成功处理的图像 ID 列表
        image_id_to_feature_idx (dict): 映射 image_id -> 图像特征索引
    Returns:
        dict: 包含 T2I 和 I2T 的 Recall@K 结果
    """
    if similarity_matrix.is_cuda:
        similarity_matrix = similarity_matrix.cpu()
        
    num_texts, num_images = similarity_matrix.shape
    results = {}

    print("Calculating Text-to-Image Recall...")
    t2i_recalls = {k: 0.0 for k in K_values}
    valid_texts_count = num_texts

    for text_idx in tqdm(range(num_texts), desc="T2I Recall"):
        correct_image_feature_idx = text_to_image_mapping[text_idx]

        sim_row = similarity_matrix[text_idx, :]

        sorted_indices = torch.argsort(sim_row, descending=True)


        rank = (sorted_indices == correct_image_feature_idx).nonzero(as_tuple=True)[0].item()


        for k in K_values:
            if rank < k: 
                t2i_recalls[k] += 1

    for k in K_values:
        t2i_recalls[k] = (t2i_recalls[k] / valid_texts_count) * 100 if valid_texts_count > 0 else 0.0
    results['T2I_Recall'] = t2i_recalls

    print("Calculating Image-to-Text Recall...")
    i2t_recalls = {k: 0.0 for k in K_values}
    valid_images_count = num_images

    for image_feature_idx in tqdm(range(num_images), desc="I2T Recall"):
        current_image_id = processed_image_ids[image_feature_idx]
        correct_text_indices = set(image_id_to_text_indices.get(current_image_id, []))

        if not correct_text_indices:
             continue

        sim_col = similarity_matrix[:, image_feature_idx]
        sorted_text_indices = torch.argsort(sim_col, descending=True)

        for k in K_values:
            top_k_text_indices = set(sorted_text_indices[:k].tolist())
            if len(correct_text_indices.intersection(top_k_text_indices)) > 0:
                 i2t_recalls[k] += 1

    for k in K_values:
        i2t_recalls[k] = (i2t_recalls[k] / valid_images_count) * 100 if valid_images_count > 0 else 0.0
    results['I2T_Recall'] = i2t_recalls

    return results


K_values_to_compute = [1, 5, 10]
recall_results = calculate_recall_hf(
    similarity_matrix,
    K_values_to_compute,
    text_to_image_mapping,
    image_id_to_text_indices,
    processed_image_ids,
    image_id_to_feature_idx
)

print("\n--- Flickr30k Recall Results (using Hugging Face dataset, batched sim) ---")
for k in K_values_to_compute:
    print(f"T2I R@{k}: {recall_results['T2I_Recall'][k]:.2f}%")
for k in K_values_to_compute:
    print(f"I2T R@{k}: {recall_results['I2T_Recall'][k]:.2f}%")

rSum = sum(recall_results['T2I_Recall'].values()) + sum(recall_results['I2T_Recall'].values())
print(f"rSum: {rSum:.2f}")

with open(f'./results/flickr30k_recall_results_{model_name.replace("/", "_")}.txt', 'w') as f:
    f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
    f.write(f"Dataset: Flickr30k\n")
    f.write(f"Model: {model_name}\n")
    # f.write(f"rSum: {rSum:.2f}\n")
    """
    复现结果（ViT-L/14@336px）：
    |  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
    |:-------:|:--------:|:-------:|
    | R@1  |   87.20%  |   67.34%|
    | R@5  |   98.90%  |   89.18%|
    | R@10  |   99.60%  |   93.58%|

    rSum: 535.80
    """
    f.write(f"复现结果（{model_name}）：\n")
    f.write("|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |\n")
    f.write("|:-------:|:--------:|:-------:|\n")
    for k in K_values_to_compute:
        f.write(f"| R@{k}  |  {recall_results['I2T_Recall'][k]:.2f}%  |   {recall_results['T2I_Recall'][k]:.2f}%   |\n")
    f.write(f"rSum: {rSum:.2f}\n")
