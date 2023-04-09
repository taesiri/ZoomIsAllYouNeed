# Zoom <img src="./assets/magnifying-glass-plus-solid.svg" width="50" height="23"> is what you need

An empirical study of the power of zoom and spatial biases in image classification

<div align="center">    

[![Website](http://img.shields.io/badge/Website-4b44ce.svg)]([https://asgaardlab.github.io/CLIPxGamePhysics/](https://taesiri.github.io/ZoomIsAllYouNeed/))
[![arXiv](https://img.shields.io/badge/arXiv-TBA-b31b1b.svg)](https://arxiv.org/abs/TBA)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/imagenet-hard)
</div>

-----

## Abstract

*Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first learn to zoom to the most discriminative region in the image and then extract features from there to predict image labels. We study six popular networks ranging from AlexNet to CLIP, and we show that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we explore the potential and limits of zoom transforms in image classification and uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zoom, we propose a state-of-the-art test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions. Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art TTA method. Additionally, we propose ImageNet-Hard, a new benchmark where zooming in alone often does not help state-of-the-art models better label images.*

## ImageNet-Hard

The ImageNet-Hard dataset is avaible to access and browser on [Hugging Face 🤗](https://huggingface.co/datasets/taesiri/imagenet-hard).

## Supplementary Material

You can find all the supplementary material on [Google Drive](https://drive.google.com/drive/folders/1bTj5GUGpGp4qssZWVuYCYbUzWy14ASJ6?usp=sharing).
