import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from module import Normalization, ContentLoss, StyleLoss
import os
class StyleTransfer:
    def __init__(self, content_img_path, style_img_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        self.imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor

        self.content_img = self.image_loader(content_img_path)
        self.style_img = self.image_loader(style_img_path)

        assert self.style_img.size() == self.content_img.size(), \
            "we need to import style and content images of the same size"
        
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
       
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def image_loader(self, image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    @staticmethod
    def imshow(tensor, title=None):
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        plt.ion()
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def get_style_model_and_losses(self, content_img, style_img, content_layers, style_layers):
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)

        print('Content_layers: ', content_layers)
        print('Style Layers:', style_layers)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        # trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img, optimizer_choice):
        print('Optimizer: ', optimizer_choice)
        if optimizer_choice == 'lbfgs':
            optimizer = optim.LBFGS([input_img.requires_grad_()])
        elif optimizer_choice == 'adam':
            optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)  # You can adjust the learning rate
        else:
            raise ValueError("Unsupported optimizer choice")
        return optimizer

    def append_to_excel(self, experiment_params):
        # Check if the Excel file exists, and create it if it doesn't
        if not os.path.isfile("experiments.xlsx"):
            # Create a new DataFrame with headers
            df = pd.DataFrame(columns=experiment_params.keys())
            # Save the DataFrame to the Excel file
            df.to_excel("experiments.xlsx", index=False)
        else:
            # Load the existing data from the Excel file
            df = pd.read_excel("experiments.xlsx")

        # Extract values from the experiment_params dictionary
        values_to_append = list(experiment_params.values())

        # Create a DataFrame with the values to append
        df_to_append = pd.DataFrame([values_to_append], columns=experiment_params.keys())

        # Append the DataFrame to the existing data (or create a new DataFrame if empty)
        df = df.append(df_to_append, ignore_index=True)

        # Save the updated DataFrame to the Excel file
        with pd.ExcelWriter("experiments.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False)
    

    def run_style_transfer(self, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], tv_weight=0.0, optimizer_choice='lbfgs', loss_choice='generic'):
        print('Building the style transfer model..')
        print('num_steps: ', num_steps)
        print('style_weight: ', style_weight)
        print('content_weight: ', content_weight)
        print('TV Weight: ', tv_weight)
        print('loss_choice: ', loss_choice)
        model, style_losses, content_losses = self.get_style_model_and_losses(content_img, style_img, content_layers, style_layers)

        input_img.requires_grad_(True)
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img, optimizer_choice)

        print('Optimizing..')
        run = [0]
        content_loss_history = []
        style_loss_history = []
        while run[0] <= num_steps:
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                # Add total variation regularization
                tv_loss = torch.sum(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])) + \
                          torch.sum(torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :]))
                tv_loss *= tv_weight

                if loss_choice == 'generic':
                    if tv_weight != 0.0:
                        loss = style_score + content_score + tv_loss  # Include TV regularization in the loss
                    else:
                        loss = style_score + content_score
                elif loss_choice == 'perceptual':
                    # Calculate the perceptual loss (VGG-style loss)
                    perceptual_loss = F.mse_loss(model(input_img), model(style_img))
                    perceptual_loss *= style_weight  # Adjust the weight as needed

                    loss = perceptual_loss + content_score  # Combine perceptual and content losses

                loss.backward()

                run[0] += 1

                if run[0] % 50 == 0:
                    print(f"run {run}:")
                    print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                    content_loss_history.append(content_score.item())
                    style_loss_history.append(style_score.item())
                    print()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.data.clamp_(0, 1)
        # Save the hyperparameters and loss results to the Excel file
        experiment_params = {
            "style_weight": style_weight,
            "content_weight": content_weight,
            "num_steps": num_steps,
            "content_layers": content_layers,
            "tv_weight": tv_weight,
            "optimizer_choice": optimizer_choice,
            "loss_choice": loss_choice,
            "content_loss": content_loss_history[-1],
            "style_loss":style_loss_history[-1],
        }
        #self.append_to_excel(experiment_params)

        return input_img

    def save_output_image(self, output, file_path="./data/images/output.jpg"):
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        output = output.cpu().clone()
        output = output.squeeze(0)
        output = unloader(output)
        output.save(file_path)

# Usage example:
# style_transfer = StyleTransfer("./data/images/dancing.jpg", "./data/images/picasso.jpg")
# output = style_transfer.run_style_transfer(style_transfer.content_img, style_transfer.style_img, style_transfer.content_img.clone())
# StyleTransfer.imshow(output, title='Output Image')
# plt.ioff()
# plt.show()
# style_transfer.save_output_image(output)
