
# Divide&Classify: Fine-Grained Classification for City-Wide Visual Geo-localization
This is the official repository of the ICCV23 paper _Divide&Classify: Fine-Grained Classification for City-Wide Visual Geo-localization_.

## Environment

The required dependencies can be installed by running 

`pip install -r requirements.txt`

to setup your python environment.

## Evaluation using pretrained models
You can run evaluation of one of our pretrained models

`$ python3 eval.py --backbone EfficientNet_B0 --resume_model path/to/model.pth --dataset_name sf_xl --train_set_path path/to/sf_xl_224/raw/train/database  --test_set_path path/to/sf_xl_224/processed/test/queries_v1`

Note that the eval script still requires the path to the train set because the classes creation is dependent on it.
We will release our pretrained models soon.

## Training

To train your own model you can run the following command

`$ python3 train.py --backbone EfficientNet_B0 --dataset_name sf_xl --train_set_path path/to/sf_xl_224/raw/train/database --val_set_path path/to/sf_xl_224/processed/test/queries_v2 --test_set_path path/to/sf_xl_224/processed/test/queries_v1`

Note that the validation set is not actually used to pick the best model.
This will train by default our AMCC classifier based on ArcFace. You can optionally use a CosFace based one by specifying `--classifier_type LMCC`, or a standard cross-entropy with `--classifier_type FC_CE`.

### Dataset 

In this work we use the SF-XL dataset, which is about 1 TB. To speed up computation we resize images from 512x512 to 224x224 resolution. To make the best use of I/O throughput and have a more efficient pipeline, we resize the images directly on disk, reducing the total dataset size to around 300 GB.
More information on the dataset can be found on the original repository [CosPlace](https://github.com/gmberton/CosPlace). To download the pre-processed version of the dataset you can specify so using this form  [_here_](https://forms.gle/wpyDzhDyoWLQygAT9).

## Acknowledgements

Parts of this repo are inspired by the following repositories:
- [CosFace](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py)
- [CosPlace](https://github.com/gmberton/CosPlace)
