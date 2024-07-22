import concurrent.futures
import os
import shutil

from script.train import (detect_DETR, detect_R_CNN, detect_RetinaNet,
                          detect_ssd, detect_yoloV8, detect_yoloV10,
                          yolo_train)

# NOTE: 学習用の写真データのパスを取得
folder_path = os.path.abspath("./data/train/photos")  # 絶対パスに変換
file_list = os.listdir(folder_path)
full_path = os.path.abspath(os.curdir)

file_path = list(
    filter(
        None,
        map(
            lambda x: os.path.join(folder_path, x) if x != ".gitkeep" else None,
            file_list,
        ),
    )
)  # os.path.joinを使用してパスを結合し、Noneをフィルタリング
# print(file_path)

# NOTE: バウンディングボックスの作成
detect_yoloV10.make_box_text_from_yoloV10(file_path, full_path)
detect_ssd.make_box_text_from_ssd(file_path, full_path)
detect_R_CNN.make_box_text_from_R_CNN(file_path, full_path)
detect_RetinaNet.make_box_text_from_RetinaNet(file_path, full_path)
detect_DETR.make_box_text_from_DETR(file_path, full_path)
detect_yoloV8.make_box_text_from_yoloV8(file_path, full_path)

print("処理が完了しました:")

# NOTE: 学習
yolo_train.yolo8_train(full_path)
print("yolo8の学習が完了しました")

weight_path = "./runs/detect/trainYolov8n/weights/best.pt"
shutil.copyfile(weight_path, "./yolov8_train.pt")
print("学習した重みをコピーしました")

# shutil.rmtree("./runs", ignore_errors=True)
# print("runsフォルダを削除しました")
