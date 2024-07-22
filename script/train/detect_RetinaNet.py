import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn
from tqdm import tqdm

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)


def make_box_text_from_RetinaNet(paths, full_path, threshold=0.5):
    transform = transforms.ToTensor()
    pl.seed_everything(0)

    model = retinanet_resnet50_fpn(pretrained=True)

    model.eval()
    for path in tqdm(paths, desc="RetinaNet"):
        img = Image.open(path)
        input = transform(img)
        output = model(input.unsqueeze(0))[0]

        image = input.permute(1, 2, 0).numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        boxes = output["boxes"].to("cuda").cpu().detach().numpy()
        labels = output["labels"].to("cuda").cpu().detach().numpy()

        if "scores" in output.keys():
            scores = output["scores"].cpu().detach().numpy()
            boxes = boxes[scores > threshold]
            labels = labels[scores > threshold]
        try:
            make_box_text_and_copy_photo(
                path,
                "RetinaNet",
                boxes[0],
                full_path,
                normalize=True,
                img_width=img.width,
                img_height=img.height,
            )
        except:
            pass
        # print(boxes[0], labels)


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
    make_box_text_from_RetinaNet(["./test.jpg"], "../..")
