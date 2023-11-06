import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, vgg_model, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:feature_layer + 1])

    def forward(self, x):
        return self.features(x)
    
#Example of usage Feature Extraction
# # Initialize VGG with the layers up to the relu4_2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg_model = models.vgg19(pretrained=True).to(device)

# # Create the feature extractor object
# feature_extractor = FeatureExtractor(vgg_model).to(device)

# # Load your images as PIL images, then preprocess them
# from PIL import Image

# # Replace 'path_to_your_image.jpg' with the actual file paths
# content_image_path = 'path_to_your_content_image.jpg'
# generated_image_path = 'path_to_your_generated_image.jpg'

# content_image = Image.open(content_image_path)
# generated_image = Image.open(generated_image_path)

# # Preprocess the images
# content_tensor = preprocess_image(content_image).to(device)
# generated_tensor = preprocess_image(generated_image).to(device)

# # Extract features
# content_features = feature_extractor(content_tensor)
# generated_features = feature_extractor(generated_tensor)

# # Compute cosine similarity
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# similarity = cos(content_features.flatten(start_dim=1),
#                  generated_features.flatten(start_dim=1))

# print(f'Feature-based similarity (cosine): {similarity.item()}')
