# ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

<div align="center">    

by [Mohammad Reza Taesiri ](https://taesiri.ai/), [Giang Nguyen](https://giangnguyen2412.github.io/), [Sarra Habchi](https://habchisarra.github.io/), [Cor-Paul Bezemer](https://asgaard.ece.ualberta.ca/), and [Anh Nguyen](https://anhnguyen.me/). 


[![Website](http://img.shields.io/badge/Website-4b44ce.svg)](https://taesiri.github.io/ZoomIsAllYouNeed/)
[![Supplementary Material](http://img.shields.io/badge/Supplementary%20Material-4b44ce.svg)](https://drive.google.com/drive/folders/1bTj5GUGpGp4qssZWVuYCYbUzWy14ASJ6?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2304.05538-b31b1b.svg)](https://arxiv.org/abs/2304.05538)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/imagenet-hard)
  [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/taesiri/imagenet-hard-4k)
</div>

-----

## Abstract

*Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first zoom to the most discriminative region in the image and then extract features from there to predict image labels, discarding the rest of the image. Studying six popular networks ranging from AlexNet to CLIP, we find that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zooming, we propose a test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions. Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art (SOTA) TTA method. We introduce ImageNet-Hard, a new benchmark that challenges SOTA classifiers including large vision-language models even when optimal zooming is allowed.*



https://user-images.githubusercontent.com/588431/231219248-08eab4cc-6c9e-4bae-8003-176149f4987c.mp4




## ImageNet-Hard

The **ImageNet-Hard** is a new benchmark that comprises an array of challenging images, curated from several validation datasets of ImageNet. This dataset challenges state-of-the-art vision models, as merely zooming in often fails to enhance their ability to correctly classify images. Consequently, even the most advanced models, such as `CLIP-ViT-L/14@336px`, struggle to perform well on this dataset, achieving only `2.02%` accuracy.

The ImageNet-Hard dataset is avaible to access and browser on Hugging Face:
- **ImageNet-Hard** [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/imagenet-hard)
- **ImageNet-Hard-4K** [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/taesiri/imagenet-hard-4k).


### Dataset Distribution

![Dataset Distribution](https://taesiri.github.io/ZoomIsAllYouNeed/static/svg/imagenet_hard_distribution.svg)



### Performance Report



| Model               | Accuracy |
| ------------------- | -------- |
| AlexNet             | 7.34     |
| VGG-16              | 12.00    |
| ResNet-18           | 10.86    |
| ResNet-50           | 14.74    |
| ViT-B/32            | 18.52    |
| EfficientNet-B0     | 16.57    |
| EfficientNet-B7     | 23.20    |
| EfficientNet-L2-Ns  | 39.00    |
| CLIP-ViT-L/14@224px | 1.86     |
| CLIP-ViT-L/14@336px | 2.02     |
| OpenCLIP-ViT-bigG-14| 15.93    |
| OpenCLIP-ViT-L-14   | 15.60    |



**Evaluation Code**

* CLIP <a target="_blank" href="https://colab.research.google.com/github/taesiri/ZoomIsAllYouNeed/blob/main/src/ImageNet_Hard/Prompt_Engineering_for_ImageNet_Hard.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
* [OpenCLIP](https://github.com/taesiri/ZoomIsAllYouNeed/blob/main/src/ImageNet_Hard/benchmark_openclip.py)
* Other models <a target="_blank" href="https://colab.research.google.com/github/taesiri/ZoomIsAllYouNeed/blob/main/src/ImageNet_Hard/Benchmark_ImageNet_Hard.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>


## Supplementary Material

You can find all the supplementary material on [Google Drive](https://drive.google.com/drive/folders/1bTj5GUGpGp4qssZWVuYCYbUzWy14ASJ6?usp=sharing).

## Citation information

If you use this software, please consider citing:

```
@article{taesiri2023zoom,
  title={ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification},
  author={Taesiri, Mohammad Reza and Nguyen, Giang and Habchi, Sarra and Bezemer, Cor-Paul and Nguyen, Anh},
  booktitle={Advances in Neural Information Processing Systems}
  year={2023}
}
```
