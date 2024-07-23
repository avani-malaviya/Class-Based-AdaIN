This is an implementation of my work on Class-Based Adaptive Instance Normalization. This work was done as an internship project at The Complex Systems Engineering Department (DISC) of ISAE-SUPAERO under the supervision of Dr. Thomas Oberlin. 

The code in this repository builds on the [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git) by [Naoto Inoue](https://github.com/naoto0804). I am sincerely grateful for the unofficial implementation as well as the [Original Implementation in Torch](https://github.com/xunhuang1995/AdaIN-style.git) by the authors of Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].

## Table of Contents

1. [Class-Based AdaIN](#class-based-adain)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Image Naming Convention](#image-naming-convention)
5. [Usage](#usage)
   - [Download & Organize Data](#download--organize-data)
   - [Download Models](#download-models)
   - [Test](#test)
   - [Train](#train)
6. [References](#references)


# Class-Based AdaIN
This work presents Class-Based Adaptive Instance Normalization (CB-AdaIN), as an approach to style transfer for domain adaptation in urban scene segmentation. We address limitations of standard AdaIN when applied to complex urban scenes by incorporating class-specific style statistics. Our method computes style statistics across multiple reference images for each semantic class, allowing for more contextually appropriate style transfer. We demonstrate CB-AdaIN's effectiveness using both a custom encoder-decoder architecture and a pre-trained Stable Diffusion VAE. Our results, show improved color adaptation and reduced artifacts compared to standard AdaIN. While the proposed method has visible limitations in generating realistic images, with improvements to the decoder architecture, CB-AdaIN shows potential to be used effectively for style transfer in urban scenes.

## Requirements
Please install requirements by `pip install -r requirements_pip.txt`. 

Alternatively, if you are using a conda environment, install requirements by `conda create --name <env> --file requirements_conda.txt`. 

This implementation uses Python 3.9 and Cuda 11.8. 

## Project Structure

The project is organized as follows:

- `input/`: Contains all input data
  - `content/`: Content images and labels
    - `GTA/`: GTA5 dataset
      - `images/`: GTA5 images
        - `train/`: Training images
        - `test/`: Test images
      - `labels_original/`: Original RGB segmentation maps
      - `labels/`: Processed grayscale labels with Cityscapes IDs
  - `style/`: Style images and labels
    - `Cityscapes/`: Cityscapes dataset
      - `images/`: Cityscapes images
        - `train/`: Training images
        - `test/`: Test images
      - `labels_original/`: Original RGB segmentation maps
      - `labels/`: Processed grayscale labels with Cityscapes IDs
- `models/`: Contains model architectures
- `utils/`: Utility scripts including data preprocessing
- `experiments/`: Saved decoder checkpoints from training
- `output/`: Saved output images from tests
- `test.py`: Script for running inference
- `train.py`: Script for training the model
- `generate_style_stats.py`: Script for generating style statistics

## Image Naming Convention

It's important to maintain a consistent naming convention for your images and their corresponding label files. Please adhere to the following convention:

1. Content images (GTA5): `[image_id].png`
   Example: `00001.png`
2. Content labels: `[image_id]_gtFine_labelIds.png`
   Example: `00001_gtFine_labelIds.png`
3. Style images (Cityscapes): `[city]_[street]_[frame_id].png`
   Example: `berlin_000000_000019.png`
4. Style labels: `[city]_[street]_[frame_id]_gtFine_labelIds.png`
   Example: `berlin_000000_000019_gtFine_labelIds.png`

Ensure that each image file has a corresponding label file with the same base name, only differing by the added `_gtFine_labelIds.png` suffix for label files.

## Usage
### Download & Organize Data
1. Download the GTA5 and Cityscapes datasets from the official websites.
2. Store the test or train images under `input/content/GTA/images/test/` or `input/content/GTA/images/train/` and `input/style/Cityscapes/images/test/` or `input/style/Cityscapes/images/train/` respectively.
3. Store the RGB labels (segmentation maps) under `input/content/GTA/labels_original/` and `input/style/Cityscapes/labels_original/` respectively.
4. Run the [script](./utils/segmap_to_cityscapes_labelIds.py) as `python /utils/segmap_to_cityscapes_labelIds.py`. The grayscale images with Cityscapes label Ids will be saved under `input/content/GTA/labels/` and `input/style/Cityscapes/labels/` respectively.

### Download models
The trained decoder can be found [here](https://drive.google.com/file/d/1xoYKg3IggCzxvewTTGBfIsEOombVHlHp/view?usp=sharing). 

The trained encoder can be downloaded from the [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git)

### Test
- To provide the content, either `--content` or `--content_dir` options can be used to provide the path to a single content image or a directory.
- To provide the style, 3 options are available:
  1. `--style`: path to a single style image
  2. `--style_dir`: path to a directory containing style images. The average of the style statistics will be taken.
  3. `--style_files`: comma separated paths to .txt files containing class based style means and stds respectively. The style files can be generated by running `python generate_style_stats.py --style_dir path/to/style/dir --style_mask_dir path/to/style/mask/dir`
- The directories containing the content and style labels must be provided using `--content_mask_dir` and `--style_mask_dir` respectively. The style labels are not required if the style files are being used. 
- `--architecture` can be assigned either `encoder-decoder` or `sd-vae` to use the standard Encoder-Decoder architecture or the Stable Diffusion (v1.5) VAE.


To recreate the final results from the report, the following code can be used. To obtain the results using the Stable-Diffusion VAE the following change must be made: `architecture="sd-vae"`

```
architecture="encoder-decoder"
content_dir="input/content/GTA/images/test/"
content_mask_dir="input/content/GTA/labels/"
style_files="multi_ref_means.txt,multi_ref_stds.txt"

python test.py --architecture $architecture --content_dir $content_dir --content_mask_dir $content_mask_dir --style_files $style_files 
```

Some other options:
* `--decoder`: Path to trained decoder.
* `--vgg`: Path to encoder.
* `--output`: Path to save output images.

For more details and parameters, please refer to --help option.

### Train
For training the arguments required are: 
1. `--content_dir`
2. `--content_mask_dir`
3. `--style_dir`
4. `--style_mask_dir`

To train the decoder for the Encoder-Decoder architecture **only** the following script can be used. The encoder is kept fixed. 
```
content_dir="input/content/GTA/images/train/"
content_mask_dir="input/content/GTA/labels/"
style_dir="input/style/cityscapes/images/train/"
style_mask_dir="input/style/cityscapes/labels/"

CUDA_VISIBLE_DEVICES=<gpu_ids> python train.py --content_dir $content_dir --content_mask_dir $content_mask_dir --style_dir $style_dir  --style_mask_dir $style_mask_dir
```
Some other options:
* `--decoder`: Path to decoder for finetuning.
*  `--save_dir`: Path to directory to save checkpoints.

For more details and parameters, please refer to --help option.


## References
- [1]: [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git) by [Naoto Inoue](https://github.com/naoto0804)
- [2]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [3]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [4]: M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. "The Cityscapes Dataset for Semantic Urban Scene Understanding." in CVPR, 2016.
- [5]: S. R. Richter*, V. Vineet*, S. Roth, and V. Koltun. "Playing for Data: Ground Truth from Computer Games." in ECCV, 2016. (*equal contribution)

## License

This project is based on the [Unofficial Pytorch Implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git) by Naoto Inoue, which is licensed under the MIT License.

The modifications and additions made for Class-Based AdaIN are also released under the MIT License.

See the [LICENSE](./LICENSE) file for full details.
