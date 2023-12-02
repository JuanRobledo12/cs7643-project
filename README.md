# CS7643-Project
This repository contains code for the CS7643 Deep Learning Project.

## CycleGAN
There are two CycleGAN notebooks that implement Cycle GAN on a reduced version of the monet2photo dataset. These notebooks also test the implementation using the following metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- Feature-Based
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance) with 192 dimensions
- ArtFID (Artwork Fréchet Inception Distance) with 192 dimensions

One notebook provides examples of single-image generation, while the other notebook evaluates the quality of generation across the content images of the dataset.

## Neural Style Transfer (NST)
- The Gatys et al. Neural Style Transfer (NST) model is implemented in `style_transfer.py` as a class that can be imported. It utilizes some modules coded in `module.py`.

- The Jupyter notebook `nst_experiments.ipynb` contains tests and examples demonstrating how to use the style transfer model and how to evaluate its outputs using the class implemented in `eval_metrics.py`.

- The evaluation process in NST follows the same metrics used in CycleGAN.

### How to use the new `style_transfer.py` with the best hyperparameters:
```
style_transfer = StyleTransfer("path_to_content_img.jpg", "path_to_style_img.jpg")

output = style_transfer.run_style_transfer(
        style_transfer.content_img,
        style_transfer.style_img,
        style_transfer.content_img.clone(),
        num_steps=300,
        style_weight=1e6,
        content_weight=0.5,
        content_layers=['relu_1'],
        style_layers=['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5'],
        tv_weight = 0.0,
        optimizer_choice='lbfgs',
        loss_choice='generic'
    )

# Add code to save or plot the result image

```


## Datasets:
- Ensure that you have the evaluation dataset to run tests on the models, particularly for monet2photo testing.
