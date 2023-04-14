# Zoom <img src="./assets/magnifying-glass-plus-solid.svg" width="50" height="23"> is what you need

An empirical study of the power of zoom and spatial biases in image classification

<div align="center">    

[![Website](http://img.shields.io/badge/Website-4b44ce.svg)](https://taesiri.github.io/ZoomIsAllYouNeed/)
[![Supplementary Material](http://img.shields.io/badge/Supplementary%20Material-4b44ce.svg)](https://drive.google.com/drive/folders/1bTj5GUGpGp4qssZWVuYCYbUzWy14ASJ6?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2304.05538-b31b1b.svg)](https://arxiv.org/abs/2304.05538)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/imagenet-hard)
</div>

-----

## Abstract

*Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first learn to zoom to the most discriminative region in the image and then extract features from there to predict image labels. We study six popular networks ranging from AlexNet to CLIP, and we show that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we explore the potential and limits of zoom transforms in image classification and uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zoom, we propose a state-of-the-art test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions. Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art TTA method. Additionally, we propose ImageNet-Hard, a new benchmark where zooming in alone often does not help state-of-the-art models better label images.*



https://user-images.githubusercontent.com/588431/231219248-08eab4cc-6c9e-4bae-8003-176149f4987c.mp4




## ImageNet-Hard

The **ImageNet-Hard** is a new benchmark that comprises an array of challenging images, curated from several validation datasets of ImageNet. This dataset challenges state-of-the-art vision models, as merely zooming in often fails to enhance their ability to correctly classify images. Consequently, even the most advanced models, such as `CLIP-ViT-L/14@336px`, struggle to perform well on this dataset, achieving only `2.02%` accuracy.

The ImageNet-Hard dataset is avaible to access and browser on  [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/imagenet-hard).


### Classifiers Performance


| Model               | Accuracy |
| ------------------- | ----- |
| ResNet-18           | 9.41  |
| ResNet-50           | 12.56 |
| ViT-B/32            | 15.95 |
| VGG19               | 10.32 |
| AlexNet             | 6.35  |
| CLIP-ViT-L/14@224px | 1.86  |
| CLIP-ViT-L/14@336px | 2.02  |
| EfficientNet-L2-Ns  | 34.23 |


**Evaluation Code**

* CLIP <a target="_blank" href="https://colab.research.google.com/github/taesiri/ZoomIsAllYouNeed/blob/main/src/ImageNet_Hard/Prompt_Engineering_for_ImageNet_Hard.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
* Other models <a target="_blank" href="https://colab.research.google.com/github/taesiri/ZoomIsAllYouNeed/blob/main/src/ImageNet_Hard/Benchmark_ImageNet_Hard.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>


## Supplementary Material

You can find all the supplementary material on [Google Drive](https://drive.google.com/drive/folders/1bTj5GUGpGp4qssZWVuYCYbUzWy14ASJ6?usp=sharing).

## Citation information


```
@article{taesiri2023zoom,
  title={Zoom is what you need: An empirical study of the power of zoom and spatial biases in image classification},
  author={Taesiri, Mohammad Reza and Nguyen, Giang and Habchi, Sarra and Bezemer, Cor-Paul and Nguyen, Anh},
  journal={arXiv preprint arXiv:2304.05538},
  year={2023}
}
```
