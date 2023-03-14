import torch
import torch.nn as nn

from utils.train_helpers import (
    tr_transforms,
    te_transforms,
    te_transforms_inc,
    common_corruptions,
)
from utils.third_party import indices_in_1k, imagenet_r_mask, objectnet_mask

import numpy as np
from functools import partial
import torchvision.transforms.functional as fv
import torchvision.transforms as transforms


def crop_at(size, slice_x, slice_y):
    def slice_crop(image, size, slice_x, slice_y):
        new_height, new_width = size, size
        width, height = image.size

        tile_size_x = width // 3
        tile_size_y = height // 3

        anchor_x = (slice_y * tile_size_x) + (tile_size_x // 2)
        anchor_y = (slice_x * tile_size_y) + (tile_size_y // 2)

        return fv.crop(
            image,
            anchor_y - (new_height // 2),
            anchor_x - (new_width // 2),
            new_height,
            new_width,
        )

    return partial(slice_crop, size=size, slice_x=slice_x, slice_y=slice_y)


def get_random_crop_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_augmix_with_random_crop_only(x_orig):

    x_processed = get_random_crop_transform()(x_orig)

    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        for _ in range(np.random.randint(1, 2)):
            x_aug = get_random_crop_transform()(x_orig)
        mix += w[i] * x_aug

    mix = m * x_processed + (1 - m) * mix
    return mix


def adapt_single(
    model, image, optimizer, criterion, corruption, niter, batch_size, prior_strength
):
    model.eval()

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    for iteration in range(niter):
        # default AugMix
        # inputs = [tr_transforms(image) for _ in range(batch_size)]  
        # RRC
        inputs = [get_random_crop_transform()(image) for _ in range(batch_size)]

        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)

        if corruption == "rendition":
            outputs = outputs[:, imagenet_r_mask]
        elif corruption == "adversarial":
            outputs = outputs[:, indices_in_1k]
        elif corruption == "imagenet1k_full":
            pass
        elif corruption == "imagenet-sketch":
            pass
        elif corruption == "objectnet":
            outputs = outputs[:, objectnet_mask]

        loss, logits = criterion(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1


def test_single(model, image, label, corruption, prior_strength):
    model.eval()

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    transform = te_transforms_inc if corruption in common_corruptions else te_transforms
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs.cuda())

        if corruption == "rendition":
            outputs = outputs[:, imagenet_r_mask]
        elif corruption == "adversarial":
            outputs = outputs[:, indices_in_1k]
        elif corruption == "imagenet1k_full":
            pass
        elif corruption == "imagenet-sketch":
            pass
        elif corruption == "objectnet":
            outputs = outputs[:, objectnet_mask]

        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence
