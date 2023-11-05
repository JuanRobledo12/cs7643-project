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

class StyleRepresentationEvaluator(nn.Module):
    def __init__(self, feature_layer=11):
        super(StyleRepresentationEvaluator, self).__init__()
        vgg_model = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg_model.children())[:feature_layer + 1])
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze the VGG model

        self.gram_matrix = GramMatrix()

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # VGG expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)  # Add batch dimension

    def forward(self, content_image_path, style_image_path):
        content_tensor = self.preprocess_image(content_image_path)
        style_tensor = self.preprocess_image(style_image_path)

        content_features = self.features(content_tensor)
        style_features = self.features(style_tensor)

        content_gram = self.gram_matrix(content_features)
        style_gram = self.gram_matrix(style_features)

        loss = nn.functional.mse_loss(content_gram, style_gram)
        return loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

# # Example usage:
# evaluator = StyleRepresentationEvaluator().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# loss = evaluator('path/to/content_image.jpg', 'path/to/style_image.jpg')

# print(f'Style Representation (Gram Matrix) Loss: {loss.item()}')

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
