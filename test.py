import torch
import clip
from PIL import Image
import os
os.environ['http_proxy'] = 'http://172.17.0.2:7890'
os.environ['https_proxy'] = 'http://172.17.0.2:7890'
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['all_proxy'] = 'socks5://172.17.0.2:7890'
os.environ['HTTP_PROXY'] = 'http://172.17.0.2:7890'
os.environ['HTTPS_PROXY'] = 'http://172.17.0.2:7890'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/home/code/CLIP/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]