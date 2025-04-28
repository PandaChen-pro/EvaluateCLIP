import torch
import clip
from PIL import Image
import os
import glob 
from tqdm import tqdm 
import time

IMAGE_DIR = "./data/test"  
# 三个模型 ViT-L/14@336px, ViT-L/14, ViT-B/32
MODEL_NAME = 'ViT-L/14@336px' 
CLASSES = ["dog", "cat"] 




device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    print(f'模型 {MODEL_NAME} 已加载到 {device}')
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()


text_prompts = [f"a photo of a {cls_name}" for cls_name in CLASSES]
print(f"生成的文本 Prompts: {text_prompts}")

with torch.no_grad():
    text_inputs = clip.tokenize(text_prompts).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(f"文本特征已编码并归一化。形状: {text_features.shape}")

correct_predictions = 0
total_images = 0
cat_correct = 0
dog_correct = 0
cat_total = 0
dog_total = 0

tp_cat = 0 
tn_cat = 0 
fp_cat = 0 
fn_cat = 0 

image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*.png')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*.jpeg')) 

if not image_files:
    print(f"错误：在目录 '{IMAGE_DIR}' 中未找到任何支持的图像文件 (.jpg, .png, .jpeg)。")
    exit()

print(f"\n开始对 '{IMAGE_DIR}' 中的 {len(image_files)} 张图片进行分类...")

for image_path in tqdm(image_files, desc="分类进度"):
    try:
        filename = os.path.basename(image_path)
        true_label = filename.split('.')[0].lower()

        if true_label not in [cls.lower() for cls in CLASSES]:
             print(f"警告：跳过文件 '{filename}'，其标签 '{true_label}' 不在定义的类别 {CLASSES} 中。")
             continue

        total_images += 1

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)


        similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=-1)
        # 使用logit_scale来获得更准确的logit值，而不是硬编码100.0 
        logit_scale = model.logit_scale.exp()
        similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        values, indices = similarity[0].topk(1)
        predicted_index = indices.item()
        predicted_label = CLASSES[predicted_index]
        confidence = values.item()

        # 打印单张图片结果
        # print(f"  文件: {filename}, 真实标签: {true_label}, 预测标签: {predicted_label}, 置信度: {confidence:.4f}")

        if predicted_label.lower() == true_label:
            correct_predictions += 1
            if true_label == 'cat':
                cat_correct += 1
            elif true_label == 'dog':
                dog_correct += 1
        else:
            print(f"\n预测错误! 文件: {filename}, 真实: {true_label}, 预测: {predicted_label}")
            try:
                image.show(title=f"错误预测 - 真实: {true_label}, 预测: {predicted_label}")
                # 可以取消下面的注释，让脚本在显示图片后暂停，按 Enter 继续
                input("按 Enter 继续...")
            except Exception as show_e:
                print(f"  无法显示图像: {show_e}")

        if true_label == 'cat':
            if predicted_label == 'cat':
                tp_cat += 1
            else: 
                fn_cat += 1
        elif true_label == 'dog':
            if predicted_label == 'dog':
                tn_cat += 1
            else: 
                fp_cat += 1

        if true_label == 'cat':
            cat_total += 1
        elif true_label == 'dog':
            dog_total += 1

    except Exception as e:
        print(f"\n处理文件 '{image_path}' 时出错: {e}")
        continue 


accuracy = (correct_predictions / total_images) * 100
cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
dog_accuracy = (dog_correct / dog_total * 100) if dog_total > 0 else 0

print(f"\n--- 分类完成 ---")
print(f"总共处理图片数: {total_images}")
print(f"正确预测数: {correct_predictions}")
print(f"Zero-Shot 分类准确率: {accuracy:.2f}%")
print(f"\n--- 各类别准确率 ---")
print(f"猫类别准确率: {cat_accuracy:.2f}% (正确: {cat_correct}/{cat_total})")
print(f"狗类别准确率: {dog_accuracy:.2f}% (正确: {dog_correct}/{dog_total})")

print(f"\n--- 混淆矩阵 ('cat' 为正类) ---")
print(f"                预测 Cat  预测 Dog")
print(f"实际 Cat       {tp_cat:<10} {fn_cat:<10}")
print(f"实际 Dog       {fp_cat:<10} {tn_cat:<10}")
print(f"TP (真阳): {tp_cat}, FN (假阴): {fn_cat}, FP (假阳): {fp_cat}, TN (真阴): {tn_cat}")

output_filename = f'./results/CatsVsDogs_Classification_{MODEL_NAME.replace("/", "_")}.txt'
with open(output_filename, 'w') as f:
    f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
    f.write(f"Dataset: CatsVsDogs\n")
    f.write(f"Model: {MODEL_NAME}\n\n")

    f.write("--- info ---\n")
    f.write(f"  Total Images: {total_images}\n")
    f.write(f"  Correct Predictions: {correct_predictions}\n")
    f.write(f"  Zero-Shot Accuracy: {accuracy:.2f}%\n")

    f.write("--- Classification Results ---\n")
    f.write(f"  Cat Accuracy: {cat_accuracy:.2f}%\n")
    f.write(f"  Dog Accuracy: {dog_accuracy:.2f}%\n")
    f.write(f"\n")

    f.write("--- Confusion Matrix ('cat' as positive class) ---\n")
    f.write(f"                预测 Cat  预测 Dog\n")
    f.write(f"实际 Cat       {tp_cat:<10} {fn_cat:<10}\n")
    f.write(f"实际 Dog       {fp_cat:<10} {tn_cat:<10}\n")
    f.write(f"TP (真阳): {tp_cat}, FN (假阴): {fn_cat}, FP (假阳): {fp_cat}, TN (真阴): {tn_cat}\n")

print(f"结果已保存到 {output_filename}")