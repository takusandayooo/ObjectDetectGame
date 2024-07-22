import os
import shutil
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)
from tqdm import tqdm

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)
# load a model pre-trained on COCO
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


def make_box_text_from_R_CNN(image_paths, full_path):
    for image_path in tqdm(image_paths, desc="R_CNN"):
        image = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        model.eval()
        with torch.no_grad():
            prediction = model([image])

        image = T.ToPILImage()(image)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        box_array = []
        for box, label, scores in zip(
            prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]
        ):
            if label == 1:
                box = list(map(float, box))
                box_array.append({"box": box, "score": scores})
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
        box_array = np.array(box_array)
        if len(box_array) == 0:
            continue
        column_max = sorted(box_array, key=lambda x: x["score"], reverse=True)[0]
        make_box_text_and_copy_photo(
            image_path,
            "R_CNN",
            column_max["box"],
            full_path,
            normalize=True,
            img_width=image.width,
            img_height=image.height,
        )


# NOTE: トレーニングデータのラベルファイルを作成し、画像をコピーする
# 正規化したい場合はnormalize=Trueにする
def make_box_text_and_copy_photo(
    image_path,
    model_name,
    bounding_box,
    full_path,
    normalize=False,
    img_width=0,
    img_height=0,
):
    if normalize:
        x_min, y_min, x_max, y_max = bounding_box
        # Calculate center coordinates
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Calculate width and height of the bounding box
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Normalize coordinates
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        box_width_norm = box_width / img_width
        box_height_norm = box_height / img_height
        bounding_box = [x_center_norm, y_center_norm, box_width_norm, box_height_norm]

        # バウンディングボックスの中心座標と幅、高さを保存
    image_filename = os.path.basename(image_path)
    filename, file_extension = os.path.splitext(
        image_filename
    )  # ファイル名と拡張子を分割
    output_file = os.path.join(
        "{}/data/train/dataSet/labels/train".format(full_path),
        os.path.splitext(os.path.basename(image_path))[0]
        + "_{}.txt".format(model_name),
    )
    with open(output_file, "w") as file:
        file.write(
            f"{0} {bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}"
        )
    new_filename = f"{filename}_{model_name}{file_extension}"  # 新しいファイル名を作成
    saved_image_path = os.path.join(
        "{}/data/train/dataSet/images/train".format(full_path),
        new_filename,  # 新しいファイル名を使用
    )
    shutil.copyfile(image_path, saved_image_path)


if __name__ == "__main__":
    make_box_text_from_R_CNN(
        ["./bus.jpg"],
        "../..",
    )
