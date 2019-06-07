# ADL2019/hw4
Conditional Image Generation

## 0. Requirements
```bash
$ pip install pytorch torchvision
$ pip install tensorflow-gpu==1.13
```

## 1. Cartoon Set
1. Download the preprocessed Cartoon Set and other resources [here](https://miulab.myDS.me:5001/sharing/S8QcGrvzt).
2. The dataset is preprocessed from [cartoonset100k.tgz](https://storage.cloud.google.com/cartoonset_public_files/cartoonset100k.tgz), licensed by Google LLC.
3. Using the [original dataset](https://google.github.io/cartoonset/download.html) is allowed.
    * Note: the original images are 500x500, RGBA
4. Preprocessed Cartoon Set Format
    * Images
        * `selected_cartoonset100k/images/`
        * 6 kinds of hair colors
        * 4 kinds of eye colors
        * 3 kinds of face colors
        * with / without glasses
        * Width, Height, Channel: 128, 128, RGB
        * Note: all the 144 (=6x4x3x2) combinations will be tested
    * Labels
        * `selected_cartoonset100k/cartoon_attr.txt`

## 2. Evaluation
### FID Score
* Both real and generated images are embedded into a **feature space** given by (a specific layer) of **Inception Net**
* Simple Baseline: **FID Score <= 180**
* Strong Baseline: **FID Score <= 120**

### Human
* See `sample_test/sample_human_testing_labels.txt`
* TAs will grade on the 144 * 5 generated images (i.e. 5 images for each of the combination of attributes)

## 3. Train Condiction GANs
```bash
$ python main.py --ckpt_dir ./checkpoints/acgan_500 --eval_dir ./eval_images/acgan_500
$ python main.py --arch resnet --ckpt_dir ./checkpoints/resnet_500 --eval_dir ./eval_images/resnet_500
$ python main.py --arch resnet --loss hinge --ckpt_dir ./checkpoints/resnet_hinge_500 --eval_dir ./eval_images/resnet_hinge_500
$ python main.py --arch resnet --ckpt_dir ./checkpoints/resnet_1000 --eval_dir ./eval_images/resnet_1000 --epochs 1000
```

## 4. Training Tips for Improvement
* [x] ACGAN with hidden concat
* [x] Resnet-based ACGAN
* [x] Adversarial Loss
    * [x] BCE Loss
    * [ ] Wasserstein (WGAN)
    * [x] Hinge Loss
* [x] Spectral Normalization
    - [torch.nn.utils.spectral_norm](https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html)
* [x] Smaller batch size + more iterations
* [x] Normalize the inputs
    - Normalize the images between -1 and 1 
    - Tanh as the last layer of the generator output
* [x] Sample Noise
    * [x] Use a Uniform distribution 
    * [ ] Use a Bernoulli distribution 
    * [ ] Use a spherical Z 
        - Don't sample from a Uniform distribution 
        - Sample from a gaussian distribution 
        - When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B 
        - Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details 
* [x] BatchNorm
    - Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images. 
    - When batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation). 
* [x] Avoid Sparse Gradients: ReLU, MaxPool
    - The stability of the GAN game suffers if you have sparse gradients
    - LeakyReLU = good (in both G and D)
    - For Downsampling, use: Average Pooling, Conv2d + stride
    - For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
      - PixelShuffle: https://arxiv.org/abs/1609.05158
* [x] Train discriminator more (sometimes) 
    - Especially when you have noise
    - Hard to find a schedule of number of D iterations vs G iterations
* [ ] [More Tips](https://github.com/soumith/ganhacks)

## 5. Evaluate Condiction GANs
```bash
$ python main.py --mode test_human --ckpt_dir ./checkpoints/acgan_500
$ mv ./results*.png ./eval_images/acgan_500
$ python main.py --mode test_fid --ckpt_dir ./checkpoints/acgan_500
$ cd FID_evaluation
$ python run_fid.py ../sample_test/fid_images
FID score: 143.087
```
```bash
$ python main.py --arch resnet --mode test_human --ckpt_dir ./checkpoints/resnet_500
$ mv ./results*.png ./eval_images/resnet_500
$ python main.py --arch resnet --mode test_fid --ckpt_dir ./checkpoints/resnet_500
$ cd FID_evaluation
$ python run_fid.py ../sample_test/fid_images
FID score: 78.082
```
```bash
$ python main.py --arch resnet --mode test_human --ckpt_dir ./checkpoints/resnet_hinge_500
$ mv ./results*.png ./eval_images/resnet_hinge_500
$ python main.py --arch resnet --mode test_fid --ckpt_dir ./checkpoints/resnet_hinge_500
$ cd FID_evaluation
$ python run_fid.py ../sample_test/fid_images
FID score: 142.319
```
```bash
$ python main.py --arch resnet --mode test_human --ckpt_dir ./checkpoints/resnet_1000
$ mv ./results*.png ./eval_images/resnet_1000
$ python main.py --arch resnet --mode test_fid --ckpt_dir ./checkpoints/resnet_1000
$ cd FID_evaluation
$ python run_fid.py ../sample_test/fid_images
FID score: 68.600
```

## 6. FID Scores

| Architecture | Loss | Epochs | FID Score | Note |
| ------------ | ---- | ------ | --------- | ---- |
| ACGAN | BCE | 500 | 143.087 | acgan_500 |
| Resnet-based ACGAN | BCE | 500 | 78.082 | resnet_500 |
| Resnet-based ACGAN | Hinge | 500 | 142.319 | resnet_hinge_500 |
| Resnet-based ACGAN | BCE | 1000 | 68.600 | resnet_1000 |

## 7. Training Progress

### ACGAN (acgan_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/acgan_500/acgan_500.gif" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_500/resnet_500.gif" width="100%">

### Resnet-based ACGAN with Hinge loss (resnet_hinge_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_hinge_500/resnet_hinge_500.gif" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_1000)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_1000/resnet_1000.gif" width="100%">

## 8. Loss and Accuracy

### ACGAN (acgan_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/acgan_500_accs.png" width="100%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/acgan_500_losses.png" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_500_accs.png" width="100%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_500_losses.png" width="100%">

### Resnet-based ACGAN with Hinge loss (resnet_hinge_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_hinge_500_accs.png" width="100%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_hinge_500_losses.png" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_1000)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_1000_accs.png" width="100%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/results/resnet_1000_losses.png" width="100%">

## 9. Human Evaluation Results

### ACGAN (acgan_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/acgan_500/results2.png" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_500/results2.png" width="100%">

### Resnet-based ACGAN with Hinge loss (resnet_hinge_500)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_hinge_500/results2.png" width="100%">

### Resnet-based ACGAN with BCE loss (resnet_1000)

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_1000/results2.png" width="100%">

###### tags: `NTU` `ADL` `2019`
