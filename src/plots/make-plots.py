#  Python code for drawing plots in the poster and slides

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)


markers = {"1 Crop": "s", "36 Crops": "X", "324 Crops": "o"}
colors = {"1 Crop": "blue", "36 Crops": "green", "324 Crops": "red"}
labels = ["AlexNet", "VGG-16", "ResNet-18", "ResNet-50", "ViT-B/32", "CLIP-ViT-L/14"]
datasets = [
    "ImageNet",
    "ImageNet-ReaL",
    "ImageNet+ReaL",
    "ImageNet-A",
    "ImageNet-R",
    "ImageNet-Sketch",
    "ObjectNet",
]


crop_1_data = np.array(
    [
        [56.16, 62.67, 61.76, 1.75, 21.10, 10.05, 14.23],
        [71.37, 78.90, 78.52, 2.69, 26.98, 16.78, 28.32],
        [69.45, 76.94, 76.47, 1.37, 32.14, 19.41, 27.59],
        [75.75, 82.63, 82.97, 0.21, 35.39, 22.91, 36.18],
        [75.75, 81.89, 82.59, 9.64, 41.29, 26.83, 30.89],
        [75.03, 80.68, 81.95, 71.28, 87.74, 58.23, 66.32],
    ]
)

crop_36_data = np.array(
    [
        [85.19, 90.30, 89.74, 31.37, 47.04, 24.40, 49.17],
        [92.30, 96.08, 95.81, 46.69, 52.86, 34.34, 62.94],
        [92.08, 95.97, 95.73, 47.48, 58.85, 37.91, 63.08],
        [94.46, 97.36, 97.40, 55.68, 61.42, 41.71, 69.60],
        [95.05, 97.61, 97.88, 68.43, 68.77, 49.10, 70.30],
        [94.19, 97.32, 97.56, 97.16, 98.60, 83.77, 89.59],
    ]
)

crop_324_data = np.array(
    [
        [90.03, 93.85, 93.48, 42.23, 55.52, 29.53, 59.65],
        [95.30, 97.90, 97.66, 58.27, 60.88, 39.90, 71.85],
        [95.15, 97.76, 97.55, 58.87, 66.89, 43.68, 71.44],
        [96.78, 98.62, 98.57, 66.68, 68.84, 47.64, 76.83],
        [97.19, 98.75, 98.91, 78.03, 75.58, 55.99, 79.28],
        [96.78, 98.69, 98.80, 98.45, 99.20, 89.00, 93.13],
    ]
)


def draw_arrows(x, y, hue, **kwargs):
    """Draw arrows from '1 Crop' to '324 Crops'."""
    x_start = x[hue == "1 Crop"].values
    y_start = y[hue == "1 Crop"].values
    x_end = x[hue == "324 Crops"].values
    y_end = y[hue == "324 Crops"].values
    for xs, ys, xe, ye in zip(x_start, y_start, x_end, y_end):
        plt.arrow(
            xs,
            ys,
            xe - xs,
            ye - ys,
            color="gray",
            length_includes_head=True,
            head_width=0.15,
            head_length=0.15,
            shape="full",
            linestyle="dashed",
        )


# Create DataFrame
data = []
for i, model in enumerate(labels):
    for j, dataset in enumerate(datasets):
        data.append([model, dataset, "1 Crop", crop_1_data[i, j]])
        data.append([model, dataset, "36 Crops", crop_36_data[i, j]])
        data.append([model, dataset, "324 Crops", crop_324_data[i, j]])

df = pd.DataFrame(data, columns=["Model", "Dataset", "Crop Condition", "Performance"])
df["Model Code"] = df["Model"].astype("category").cat.codes

df["Model"] = pd.Categorical(df["Model"], categories=labels, ordered=True)
df["Model Code"] = df["Model"].cat.codes

df_1_324 = df[df["Crop Condition"].isin(["1 Crop", "324 Crops"])].copy()
df_1_324["Row"] = "1 and 324 Crops"
df["Row"] = "All Crops"

df_combined = pd.concat([df_1_324, df])


def generate_plots(row_filter, include_baseline=False):
    subset_data = df_combined[df_combined["Row"] == row_filter]
    for dataset in datasets:
        data = subset_data[subset_data["Dataset"] == dataset]
        g = sns.FacetGrid(data, height=8, ylim=(0, 100), aspect=1)

        if include_baseline:
            g.map(plt.axhline, y=90, color="green", linestyle="--", linewidth=5)

        g.map(
            sns.scatterplot,
            "Model Code",
            "Performance",
            "Crop Condition",
            palette=colors,
            s=500,
            marker="o",
            legend=False,
        )
        g.map(draw_arrows, "Model Code", "Performance", "Crop Condition")

        g.set_axis_labels("Model", "Performance")
        g.set_titles("{col_name}")
        g.fig.suptitle(dataset, fontsize=24)

        for ax in g.axes.flat:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels(range(0, 101, 20))

        suffix = "_baseline" if include_baseline else ""
        g.savefig(f"{row_filter}_{dataset}{suffix}.png", bbox_inches="tight", dpi=300)
        plt.close(g.fig)


generate_plots("1 and 324 Crops", include_baseline=True)
generate_plots("1 and 324 Crops", include_baseline=False)
generate_plots("All Crops", include_baseline=True)
generate_plots("All Crops", include_baseline=False)

