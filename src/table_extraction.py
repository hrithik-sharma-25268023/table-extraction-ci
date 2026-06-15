"""Module for table detection and visualization"""

from typing import Tuple
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

MODEL = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
PROCESSOR = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")


def table_detection(image_path: str) -> Tuple:
    """table detection method"""

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((int(width * 0.5), int(height * 0.5)))
    inputs = PROCESSOR(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = MODEL(**inputs)
    results = PROCESSOR.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(image.size[1], image.size[0])])[0]
    return image, results


def save_results(pil_img: Image, scores: torch.Tensor,
                 labels: torch.Tensor, boxes:torch.Tensor, destination_path: str) -> None:
    """saves the results of the images"""

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, box, color in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        xmin, ymin, xmax, ymax = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        text = f"{MODEL.config.id2label[label]}: {score:.2f}"
        ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")


if __name__ == "__main__":
    image_path = "/home/hrithik-dev/ds-labs/table-extraction-ci/data/images/American-Express-Annual-Report-2025.pdf/page81.jpg"

    image, results = table_detection(image_path)
    save_results(image, results["scores"], results["labels"], results["boxes"])
