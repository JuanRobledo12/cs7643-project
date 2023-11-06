eval_metrics.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import numpy as np
import cv2

class EvaluationMetrics:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.vgg_model = models.vgg19(pretrained=True).to(self.device)
        self.feature_extractor = FeatureExtractor(self.vgg_model).to(self.device)
    
    def compute_ssim(self, content_image_path, stylized_image_path):
        content_image = cv2.imread(content_image_path)
        stylized_image = cv2.imread(stylized_image_path)

        # Resize the stylized image to match the content image's dimensions
        stylized_image = cv2.resize(stylized_image, (content_image.shape[1], content_image.shape[0]))

        content_gray = cv2.cvtColor(content_image, cv2.COLOR_BGR2GRAY)
        stylized_gray = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2GRAY)

        # Now compute SSIM
        score, _ = compare_ssim(content_gray, stylized_gray, full=True)
        return score


    def compute_psnr(self, content_image_path, stylized_image_path):
        content_image = cv2.imread(content_image_path)
        stylized_image = cv2.imread(stylized_image_path)
        # Resize the stylized image to match the content image's dimensions
        stylized_image = cv2.resize(stylized_image, (content_image.shape[1], content_image.shape[0]))
        score = compare_psnr(content_image, stylized_image)
        return score

    def compute_feature_similarity(self, content_tensor, generated_tensor):
        content_features = self.feature_extractor(content_tensor)
        generated_features = self.feature_extractor(generated_tensor)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(content_features.flatten(start_dim=1),
                         generated_features.flatten(start_dim=1))
        return similarity.item()

    def compute_lpips_and_artFID(self, content_tensor, generated_tensor):
        d = self.loss_fn_alex(content_tensor, generated_tensor)
        art_fid = (1 + d.item())*(1 + 4.481046369811025)
        return (d.item(), art_fid)

    # Helper functions
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0).to(self.device)  # Add batch dimension

# Feature Extractor class as used in the example provided
class FeatureExtractor(nn.Module):
    def __init__(self, vgg_model, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:feature_layer + 1])

    def forward(self, x):
        return self.features(x)

# # Usage:
# eval_metrics = EvaluationMetrics(device='cuda')

# # Replace 'path_to_content_image.jpg' and 'path_to_stylized_image.jpg' with the actual file paths
# content_image_path = 'path_to_content_image.jpg'
# generated_image_path = 'path_to_stylized_image.jpg'

# # Compute SSIM
# ssim_score = eval_metrics.compute_ssim(content_image_path, generated_image_path)
# print(f'SSIM: {ssim_score}')

# # Compute PSNR
# psnr_score = eval_metrics.compute_psnr(content_image_path, generated_image_path)
# print(f'PSNR: {psnr_score}')

# # Preprocess the images for feature-based similarity and LPIPS
# content_tensor = eval_metrics.preprocess_image(content_image_path)
# generated_tensor = eval_metrics.preprocess_image(generated_image_path)

# # Compute feature-based similarity
# feature_similarity = eval_metrics.compute_feature_similarity(content_tensor, generated_tensor)
# print(f'Feature-based similarity (cosine): {feature_similarity}')

# # Compute LPIPS
# lpips_score = eval_metrics.compute_lpips(content_tensor, generated_tensor)
# print(f'LPIPS: {lpips_score}')
