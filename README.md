# cs7643-project
Code for the CS7643 Deep Learning Project


## CycleGAN
There are two cycleGAN notebooks with the implementation of cycle GAN on the monet2photo dataset and it is also test it on the following metrics:
* SSIM
* PSNR
* Feature-Based
* lpips
* FID (192 dim)
* ArtFID (192 dim)

## Neural Style Transfer (NST)
* The Gatys et al. NST model is implemented in `style_transfer.py` and it is a class that can be imported. It makes use of some modules coded in `module.py`.

* The jupyter notebook `nst_experiments.ipynb` has tests and examples on how to use the style transfer model and how to evaluate it outputs using the class implemented in `eval_metrics.py`

* The evaluation follows the same metrics as CycleGAN.


## Datasets:
* Make sure to have the eval dataset to run monet2photo tets on the models.