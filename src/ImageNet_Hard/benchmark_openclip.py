import open_clip
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import requests
from torch.utils.data import DataLoader
from datasets import load_dataset
import csv

from prettytable import PrettyTable

sns.set_style("whitegrid")

# Define templates
imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

url = "https://gist.githubusercontent.com/taesiri/5b5edb5452f2f20d82d5ed1bb58ab574/raw/0376003d3999799f99208fb338cfd06ee44372b5/imagenet-labels.json"

response = requests.get(url)

if response.status_code == 200:
    imagenet_classes = response.json()


# Define helper functions
def apply_transforms(examples, preprocess):
    examples["pixel_values"] = examples["image"]
    examples["image"] = [preprocess(image) for image in examples["image"]]
    return examples


def zeroshot_classifier(classnames, templates, model, tokenizer):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = tokenizer(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)

            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def collate_fn(batch):
    labels = [item["label"] for item in batch]
    labels = [label + [-1] * (10 - len(label)) for label in labels]

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor(labels),
    }


# Define benchmark function
def run_benchmark(
    model, tokenizer, preprocess, imagenet_classes, imagenet_templates, dataset
):
    zeroshot_weights = zeroshot_classifier(
        imagenet_classes, imagenet_templates, model, tokenizer
    )

    loader = DataLoader(dataset, batch_size=32, num_workers=2, collate_fn=collate_fn)
    correct_ones = 0
    with torch.no_grad():
        n = 0.0
        for i, (batch) in enumerate(tqdm(loader)):
            images, target = batch["image"], batch["label"]
            images = images.cuda()
            target = target.cuda()

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights

            pred = logits.data.max(1)[1]
            correct_ones += (pred[:, None] == target).any(1).sum().item()

            n += images.size(0)

    top1 = 100 * correct_ones / n
    return top1


def main():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(device=0))
    print(torch.__version__)

    # Define a list of model configurations to be benchmarked
    models = [
        ("RN50", "yfcc15m"),
        ("RN50", "cc12m"),
        ("RN50-quickgelu", "yfcc15m"),
        ("RN50-quickgelu", "cc12m"),
        ("RN101", "yfcc15m"),
        ("RN101-quickgelu", "yfcc15m"),
        ("ViT-B-32", "laion400m_e31"),
        ("ViT-B-32", "laion400m_e32"),
        ("ViT-B-32", "laion2b_e16"),
        ("ViT-B-32", "laion2b_s34b_b79k"),
        ("ViT-B-32", "datacomp_m_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_clip_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_laion_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_image_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_text_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_basic_s128m_b4k"),
        ("ViT-B-32", "commonpool_m_s128m_b4k"),
        ("ViT-B-32", "datacomp_s_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_clip_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_laion_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_image_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_text_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_basic_s13m_b4k"),
        ("ViT-B-32", "commonpool_s_s13m_b4k"),
        ("ViT-B-32-quickgelu", "laion400m_e31"),
        ("ViT-B-32-quickgelu", "laion400m_e32"),
        ("ViT-B-16", "laion400m_e31"),
        ("ViT-B-16", "laion400m_e32"),
        ("ViT-B-16", "laion2b_s34b_b88k"),
        ("ViT-B-16", "datacomp_l_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_clip_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_laion_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_image_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_text_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_basic_s1b_b8k"),
        ("ViT-B-16", "commonpool_l_s1b_b8k"),
        ("ViT-B-16-plus-240", "laion400m_e31"),
        ("ViT-B-16-plus-240", "laion400m_e32"),
        ("ViT-L-14", "laion400m_e31"),
        ("ViT-L-14", "laion400m_e32"),
        ("ViT-L-14", "laion2b_s32b_b82k"),
        ("ViT-L-14", "datacomp_xl_s13b_b90k"),
        ("ViT-L-14", "commonpool_xl_clip_s13b_b90k"),
        ("ViT-L-14", "commonpool_xl_laion_s13b_b90k"),
        ("ViT-L-14", "commonpool_xl_s13b_b90k"),
        ("ViT-H-14", "laion2b_s32b_b79k"),
        ("ViT-g-14", "laion2b_s12b_b42k"),
        ("ViT-g-14", "laion2b_s34b_b88k"),
        ("ViT-bigG-14", "laion2b_s39b_b160k"),
        ("roberta-ViT-B-32", "laion2b_s12b_b32k"),
        ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
        ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
        ("convnext_base", "laion400m_s13b_b51k"),
        ("convnext_base_w", "laion2b_s13b_b82k"),
        ("convnext_base_w", "laion2b_s13b_b82k_augreg"),
        ("convnext_base_w", "laion_aesthetic_s13b_b82k"),
        ("convnext_base_w_320", "laion_aesthetic_s13b_b82k"),
        ("convnext_base_w_320", "laion_aesthetic_s13b_b82k_augreg"),
        ("convnext_large_d", "laion2b_s26b_b102k_augreg"),
        ("convnext_large_d_320", "laion2b_s29b_b131k_ft"),
        ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup"),
        ("convnext_xxlarge", "laion2b_s34b_b82k_augreg"),
        ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_rewind"),
        ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),
        ("coca_ViT-B-32", "laion2b_s13b_b90k"),
        ("coca_ViT-B-32", "mscoco_finetuned_laion2b_s13b_b90k"),
        ("coca_ViT-L-14", "laion2b_s13b_b90k"),
        ("coca_ViT-L-14", "mscoco_finetuned_laion2b_s13b_b90k"),
    ]

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    imagenet_hard_dataset = load_dataset("taesiri/imagenet-hard", split="validation")

    results = []

    for model_name, pretrained in models:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

            tokenizer = open_clip.get_tokenizer(model_name)

            model = model.cuda()

            imagenet_hard_dataset.set_transform(
                lambda examples: apply_transforms(examples, preprocess)
            )

            top1 = run_benchmark(
                model,
                tokenizer,
                preprocess,
                imagenet_classes,
                imagenet_templates,
                imagenet_hard_dataset,
            )

            # Add the results to the table
            results.append([model_name, pretrained, f"{top1:.2f}"])
        except Exception as e:
            print(f"Error: {e} - {model_name} - {pretrained}")

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=["Model", "Pretrained", "Top-1 Accuracy"])

    # Create a pretty table to display the results
    results_table = PrettyTable()
    results_table.field_names = ["Model", "Pretrained", "Top-1 Accuracy"]
    for result in results:
        results_table.add_row(result)

    print(results_table)

    # Save DataFrame to CSV
    df.to_csv("openclip_results.csv", index=False)


if __name__ == "__main__":
    main()
