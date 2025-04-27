import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import os # 已经导入过一次了

# 设置代理 (如果需要的话)
os.environ['http_proxy'] = 'http://172.17.0.2:7890'
os.environ['https_proxy'] = 'http://172.17.0.2:7890'

# 添加特征保存路径
FEATURES_DIR = "./saved_features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# 设置设备和模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'ViT-B/32' # 或 ViT-L/14 等
# 考虑显存，ViT-B/32 应该比较合适
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

# 尝试加载已保存的特征
features_dict = load_features()

if features_dict is None:
    print("未找到保存的特征，开始提取特征...")
    print("Loading Flickr30k dataset from Hugging Face...")
    try:
        # 增加 cache_dir 参数
        hf_dataset = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True, cache_dir="./hf_cache")
        print("Dataset loaded successfully.")
        print(f"Test set size: {len(hf_dataset)}")
        print("Dataset features:", hf_dataset.features)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you are logged in to Hugging Face Hub if required ('huggingface-cli login')")
        print("Or check the dataset name and availability.")
        exit()

    # Lists to collect features (on CPU initially)
    image_features_list = []
    text_features_list = []

    # Mappings and metadata
    processed_image_ids = []
    all_texts = []
    text_to_image_mapping = [] # Maps text index to image feature index
    image_id_to_feature_idx = {} # Maps image_id to image feature index
    image_id_to_text_indices = {} # Maps image_id to list of text indices

    print("Extracting image features...")
    current_image_idx = 0
    with torch.no_grad():
        for i, item in enumerate(tqdm(hf_dataset, desc="Processing Images")):
            image = item['image']
            image_id = item.get('image_id', str(i))
            if image.mode != "RGB":
                image = image.convert("RGB")
            try:
                # Process image one by one on the device
                image_input = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_list.append(image_features.cpu()) # Move to CPU immediately
                processed_image_ids.append(image_id)
                image_id_to_feature_idx[image_id] = current_image_idx
                current_image_idx += 1
            except Exception as e:
                print(f"Error processing image {image_id} (index {i}): {e}")
                # Optionally skip adding image_id to mappings if processing failed
                continue

    print("Preparing text data and extracting text features...")
    current_text_idx = 0
    # Extract text features in batches
    text_batch_size = 64 # Adjust text batch size for encoding if needed, 4 might be too small but safe
    current_batch_texts = []

    for i, item in enumerate(tqdm(hf_dataset, desc="Preparing Texts & Encoding")):
        image_id = item.get('image_id', str(i))

        # Check if the corresponding image was processed successfully
        if image_id not in image_id_to_feature_idx:
            continue # Skip texts for images that failed

        captions = item['caption']
        image_feature_idx = image_id_to_feature_idx[image_id]

        if image_id not in image_id_to_text_indices:
            image_id_to_text_indices[image_id] = []

        for caption in captions:
            all_texts.append(caption)
            text_to_image_mapping.append(image_feature_idx) # Record image index for this text
            image_id_to_text_indices[image_id].append(current_text_idx) # Record text index for this image_id
            current_batch_texts.append(caption)
            current_text_idx += 1

            # Process text batch when full
            if len(current_batch_texts) >= text_batch_size:
                 with torch.no_grad():
                    # `truncate=True` handles long texts
                    text_inputs = clip.tokenize(current_batch_texts, truncate=True).to(device)
                    text_features = model.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features_list.append(text_features.cpu()) # Move to CPU immediately
                 current_batch_texts = [] # Clear the batch

    # Process any remaining texts in the last batch
    if current_batch_texts:
         with torch.no_grad():
            text_inputs = clip.tokenize(current_batch_texts, truncate=True).to(device)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features.cpu())

    # Concatenate all features collected on CPU
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

    # Save the extracted features
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


# --- 计算相似度矩阵 (分批处理) ---
# 不再直接将所有特征移到GPU进行大矩阵乘法
num_texts = text_features_all.shape[0]
num_images = image_features_all.shape[0]
embedding_dim = text_features_all.shape[1] # Should be 512 for ViT-B/32

# Initialize similarity matrix on CPU
similarity_matrix = torch.zeros(num_texts, num_images, dtype=image_features_all.dtype) # Use the same dtype

print(f"Initialized similarity matrix shape: {similarity_matrix.shape} on CPU")

# Define batch sizes for similarity calculation
# Adjust these based on your GPU memory. Start small and increase if possible.
# A 1080 (8GB) should be able to handle batches that result in ~2-4GB intermediate tensors.
# 5000 * 1000 matrix, size 5000*1000*4 bytes = 20MB
# Batching strategy: Compute blocks of the similarity matrix
# text_batch_size_sim: number of texts per batch
# image_batch_size_sim: number of images per batch
# Computing (text_batch_size_sim, embed) @ (embed, image_batch_size_sim)
# Intermediate size on GPU will be roughly (text_batch_size_sim * image_batch_size_sim * 4 bytes) + input tensors
# Let's try batching texts and calculating sim with *all* images first if that fits
# If not, we'll need to batch images too.

# Option 1: Batch Text, calculate sim with ALL Images
# This requires image_features_all (1000*512*4 bytes = ~2MB) + text_batch (batch*512*4)
# + result batch (batch*1000*4) + intermediate workspace on GPU.
# Let's try text_batch_size_sim = 1000? 1000*1000*4 bytes = 4MB result batch.
# Try text batch sizes that make the result batch plus inputs fit.
text_batch_size_sim = 256 # Example batch size for text in similarity calculation

print("Calculating similarity matrix in batches...")

# Move image features to GPU once (if they fit, 1000x512 is usually fine)
image_features_gpu = image_features_all.to(device)

with torch.no_grad():
    for i in tqdm(range(0, num_texts, text_batch_size_sim), desc="Calculating Similarity"):
        text_batch_start = i
        text_batch_end = min(i + text_batch_size_sim, num_texts)
        
        # Get a batch of text features and move to GPU
        text_features_batch = text_features_all[text_batch_start:text_batch_end, :].to(device)
        
        # Calculate similarity for this text batch against all images
        # (batch_size, embed) @ (embed, num_images) -> (batch_size, num_images)
        batch_similarity = (text_features_batch @ image_features_gpu.T)
        
        # Move the result batch back to CPU and place it in the main matrix
        similarity_matrix[text_batch_start:text_batch_end, :] = batch_similarity.cpu()

print("Similarity matrix calculation complete.")
print(f"Final similarity matrix shape: {similarity_matrix.shape} on CPU")

# --- 6. 计算召回率 (Recall@K) ---
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
    # Ensure similarity_matrix is on CPU for this function
    if similarity_matrix.is_cuda:
        similarity_matrix = similarity_matrix.cpu()
        
    num_texts, num_images = similarity_matrix.shape
    results = {}

    # --- Text-to-Image Recall (文本检索图像) ---
    print("Calculating Text-to-Image Recall...")
    t2i_recalls = {k: 0.0 for k in K_values}
    valid_texts_count = num_texts

    for text_idx in tqdm(range(num_texts), desc="T2I Recall"):
        correct_image_feature_idx = text_to_image_mapping[text_idx]

        # Get similarity row for current text
        sim_row = similarity_matrix[text_idx, :]
        # Find the rank of the correct image
        # torch.argsort gives indices of sorted values
        sorted_indices = torch.argsort(sim_row, descending=True)

        # Find the position (rank) of the correct_image_feature_idx in the sorted indices
        # The rank is 0-indexed, so rank 0 means it's the top prediction
        rank = (sorted_indices == correct_image_feature_idx).nonzero(as_tuple=True)[0].item()

        # Check R@K
        for k in K_values:
            if rank < k: # If rank is 0, 1, ..., k-1, it's within top K
                t2i_recalls[k] += 1

    for k in K_values:
        t2i_recalls[k] = (t2i_recalls[k] / valid_texts_count) * 100 if valid_texts_count > 0 else 0.0
    results['T2I_Recall'] = t2i_recalls

    # --- Image-to-Text Recall (图像检索文本) ---
    print("Calculating Image-to-Text Recall...")
    i2t_recalls = {k: 0.0 for k in K_values}
    valid_images_count = num_images

    for image_feature_idx in tqdm(range(num_images), desc="I2T Recall"):
        current_image_id = processed_image_ids[image_feature_idx]
        # Get the list of correct text indices for this image
        correct_text_indices = set(image_id_to_text_indices.get(current_image_id, []))

        if not correct_text_indices:
             # This image didn't have any corresponding texts in the processed set
             # Should not happen if data preparation was correct and all texts corresponding to processed images are included
             continue

        # Get similarity column for current image
        sim_col = similarity_matrix[:, image_feature_idx]
        # Find the rank of all texts for this image
        sorted_text_indices = torch.argsort(sim_col, descending=True)

        # Check if any of the top K texts are correct
        for k in K_values:
            top_k_text_indices = set(sorted_text_indices[:k].tolist())
            if len(correct_text_indices.intersection(top_k_text_indices)) > 0:
                 i2t_recalls[k] += 1

    for k in K_values:
        i2t_recalls[k] = (i2t_recalls[k] / valid_images_count) * 100 if valid_images_count > 0 else 0.0
    results['I2T_Recall'] = i2t_recalls

    return results


# --- 执行计算 ---
K_values_to_compute = [1, 5, 10]
recall_results = calculate_recall_hf(
    similarity_matrix,
    K_values_to_compute,
    text_to_image_mapping,
    image_id_to_text_indices,
    processed_image_ids,
    image_id_to_feature_idx
)
# --- 打印结果 ---
print("\n--- Flickr30k Recall Results (using Hugging Face dataset, batched sim) ---")
for k in K_values_to_compute:
    print(f"T2I R@{k}: {recall_results['T2I_Recall'][k]:.2f}%")
for k in K_values_to_compute:
    print(f"I2T R@{k}: {recall_results['I2T_Recall'][k]:.2f}%")

rSum = sum(recall_results['T2I_Recall'].values()) + sum(recall_results['I2T_Recall'].values())
print(f"rSum: {rSum:.2f}")
