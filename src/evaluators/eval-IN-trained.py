from PIL import Image, ImageChops
import argparse
import os
import pickle
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.transforms.functional as fv
from functools import partial
import torchvision.models as models

random.seed(42)
np.random.seed(42)

print("Torch Version:", torch.__version__)
print("CUDA  Version:", torch.version.cuda)
print("Using device:", torch.cuda.get_device_name(device=0))

IMAGENET_TRAIN_FOLDER = "/home/mohammad/dataset/ILSVRC2012_img_train/"

model_bank = ["vgg16", "alexnet", "resnet18", "resnet50", "vit_b_32"]


def crop_at2(size, slice_x, slice_y):
    def trim(im):
        bg = Image.new(im.mode, im.size, (0, 0, 0, 0))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return im

    def slice_crop2(image, size, slice_x, slice_y):
        new_height, new_width = size, size
        width, height = image.size

        tile_size_x = width / 3
        tile_size_y = height / 3

        anchor_x = (slice_y * tile_size_x) + (tile_size_x / 2)
        anchor_y = (slice_x * tile_size_y) + (tile_size_y / 2)

        cropped_image = fv.crop(
            image,
            anchor_y - (new_height / 2),
            anchor_x - (new_width / 2),
            new_height,
            new_width,
        )

        trimmed = trim(cropped_image)

        return fv.pad(
            trimmed,
            padding=[
                (224 - trimmed.size[0]) // 2,
                (224 - trimmed.size[1]) // 2,
                (224 - trimmed.size[0]) // 2,
                (224 - trimmed.size[1]) // 2,
            ],
            padding_mode="constant",
        )

    return partial(slice_crop2, size=size, slice_x=slice_x, slice_y=slice_y)


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


def generate_transform_size_list():
    tsize_list = [
        10,
        16,
        32,
        48,
        64,
        96,
        128,
        192,
        224,
        240,
        256,
        288,
        320,
        384,
        448,
        512,
        576,
        640,
        664,
        672,
        680,
        690,
        700,
        720,
        768,
        832,
        896,
    ]
    tsize_list = tsize_list + [int(x) for x in np.linspace(10, 1024, 10)]

    return tsize_list


def benchmark_masked(model, dataset_folder, img_transform, all_wnids, bs=16):
    model = model.cuda()
    model.eval()

    predictions = []
    confidences = []

    dataset = ImageFolder(root=dataset_folder, transform=img_transform)
    test_loader = DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True
    )

    dataset_wnids = set([x[0].split("/")[-2] for x in test_loader.dataset.imgs])
    mask = [wnid in set(dataset_wnids) for wnid in all_wnids]

    correct_ones = 0
    with torch.inference_mode():
        for _, (data, target) in enumerate(tqdm(test_loader)):
            data = data.cuda()
            target = target.cuda()

            output = model(data)[:, mask]

            pred = output.data.max(1)[1]
            correct_ones += pred.eq(target.data).sum().item()
            probs = F.softmax(output, dim=1)

            predictions.extend(pred.data.cpu().numpy())
            confidences.extend(probs.data.cpu().numpy())

    del data
    del target
    torch.cuda.empty_cache()

    return 100 * correct_ones / len(dataset), predictions, confidences


# TODO: refine the code here
def run(model_name, VAL_FOLDER, OUTPUT, slice_x, slice_y, output_dir):
    all_wnids = list(sorted(os.listdir(IMAGENET_TRAIN_FOLDER)))

    list_of_tsizes = generate_transform_size_list()
    confidences = {}
    predictions = {}
    accuracies = {}

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(pretrained=True)
    else:
        raise

    for tsize in tqdm(list_of_tsizes):
        current_transform = transforms.Compose(
            [
                transforms.Resize(
                    tsize,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=None,
                ),
                crop_at(224, slice_x, slice_y),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        acc, preds, confs = benchmark_masked(
            model,
            dataset_folder=VAL_FOLDER,
            img_transform=current_transform,
            all_wnids=all_wnids,
        )
        accuracies[tsize] = acc
        confidences[tsize] = confs
        predictions[tsize] = preds

    if not os.path.exists(f"./{output_dir}/plot_pytorch_overallping"):
        os.mkdir(f"./{output_dir}/plot_pytorch_overallping")

    if not os.path.exists(f"./{output_dir}/accuracies_pytorch_overallping"):
        os.mkdir(f"./{output_dir}/accuracies_pytorch_overallping")

    if not os.path.exists(f"./{output_dir}/predictions_pytorch_overallping"):
        os.mkdir(f"./{output_dir}/predictions_pytorch_overallping")

    plt.savefig(
        f"./{output_dir}/plot_pytorch_overallping/{OUTPUT}_{slice_x}_{slice_y}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"./{output_dir}/plot_pytorch_overallping/{OUTPUT}_{slice_x}_{slice_y}.png",
        bbox_inches="tight",
    )

    with open(
        f"./{output_dir}/accuracies_pytorch_overallping/{OUTPUT}_{slice_x}_{slice_y}.pickle",
        "wb",
    ) as f:
        pickle.dump(accuracies, f)

    with open(
        f"./{output_dir}/predictions_pytorch_overallping/{OUTPUT}_{slice_x}_{slice_y}.pickle",
        "wb",
    ) as f:
        pickle.dump(predictions, f)


def main():
    parser = argparse.ArgumentParser(description="Zoom in Benchmark")
    parser.add_argument(
        "--val", help="Validation set root dir", type=str, required=True
    )

    parser.add_argument("--slice_x", help="Zoom location", type=int, required=True)
    parser.add_argument("--slice_y", help="Zoom location", type=int, required=True)
    parser.add_argument(
        "--output_dir", help="Output directory", type=str, required=True
    )

    val_folder = parser.parse_args().val
    slice_x = parser.parse_args().slice_x
    slice_y = parser.parse_args().slice_y
    output_dir = parser.parse_args().output_dir

    os.makedirs(output_dir, exist_ok=True)

    for model_id in tqdm(reversed(range(len(model_bank)))):
        selected_model = model_bank[model_id]
        val_foldername = os.path.basename(os.path.normpath(val_folder))
        outout_name = f"{val_foldername}-{selected_model}-{slice_x}-{slice_y}"

        print(
            f"Running inference for :{val_folder}, with {selected_model} model, saving to {outout_name}"
        )

        run(
            selected_model,
            val_folder,
            outout_name,
            slice_x,
            slice_y,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
