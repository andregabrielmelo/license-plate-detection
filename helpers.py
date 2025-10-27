from typing import Tuple
import matplotlib.pyplot as plt
import cv2

from models.label import Label


def draw_bboxes(
    img_rgb: cv2.typing.MatLike,
    labels: list[Label],
    color: Tuple,
    thickness: int = 2,
) -> cv2.typing.MatLike:
    img_copy = img_rgb.copy()
    h, w, _ = img_copy.shape

    for lbl in labels:
        cx = lbl.center_x
        cy = lbl.center_y
        bw = lbl.width
        bh = lbl.height

        # Converter coordenadas YOLO (cx,cy,bw,bh) -> (x_min,y_min,x_max,y_max)
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)

        # Desenhar a bounding box
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)

    return img_copy


def debug_img(images):
    ncols = len(images)
    f, axarr = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Intermediate steps from localize_char_bbox
    for i, step in enumerate(images):
        axarr[i].imshow(
            step["image"], cmap="gray" if len(step["image"].shape) == 2 else None
        )
        axarr[i].set_title(step["title"])
        axarr[i].axis("off")

    plt.tight_layout()
    plt.show()


def debug_img_individual(images):
    """
    Display each image individually with its title.

    Parameters:
        images (list of dict): List of dictionaries containing
            {
                "image": <numpy.ndarray>,
                "title": <str>
            }
    """
    for step in images:
        plt.figure(figsize=(5, 5))
        plt.imshow(
            step["image"], cmap="gray" if len(step["image"].shape) == 2 else None
        )
        plt.title(step.get("title", ""))
        plt.axis("off")
        plt.show()
