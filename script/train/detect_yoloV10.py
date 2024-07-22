import os
import shutil
import warnings

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)


def make_box_text_from_yoloV10(paths, full_path):
    model = YOLO("{}/yolov10x.pt".format(full_path))
    model = model.to("cuda")
    for path in tqdm(paths, desc="yoloV10"):
        im1 = Image.open(path)
        results = model.predict(source=im1, verbose=False)
        items = results[0]
        for item in items:
            cls = int(item.boxes.cls)  # cls, (N, 1)
            score = item.boxes.conf.cpu().numpy()[0]  # confidence score, (N, 1)
            x1, y1, x2, y2 = item.boxes.xyxy.cpu().numpy()[0]
            # print(cls, score, x1, y1, x2, y2)
            if cls == 0:
                make_box_text_and_copy_photo(
                    path,
                    "YoloV10",
                    [x1, y1, x2, y2],
                    full_path,
                    normalize=True,
                    img_width=im1.width,
                    img_height=im1.height,
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
    # NOTE: 見やすいように改行


if __name__ == "__main__":
    make_box_text_from_yoloV10(["test.jpg"], "../..")
