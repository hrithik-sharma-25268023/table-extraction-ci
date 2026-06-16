"""table extraction and detection module"""

from typing import Tuple
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


@st.cache_resource
def load_model() -> None:
    """model and processor"""

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    return model, processor


def table_detection(image_path: str) -> Tuple:
    """table detection method"""

    model, processor = load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(image.size[1], image.size[0])])[0]

    return image, results


def save_results(pil_img: Image, scores: torch.Tensor, labels: torch.Tensor,
                 boxes: torch.Tensor, destination_path: str) -> None:

    model, _ = load_model()
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(pil_img)
    colors = COLORS * 100
    for score, label, box, color in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        xmin, ymin, xmax, ymax = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                     fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f"{model.config.id2label[label]}: {score:.2f}", fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.5))
    ax.axis("off")
    fig.savefig(destination_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
