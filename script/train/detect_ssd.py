import os
import shutil
import warnings

import matplotlib.patches as patches
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)


def make_box_text_from_ssd(image_paths, full_path, confidence_threshold=0.4):
    # モデルの読み込み
    precision = "fp32"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # デバイスを指定
    ssd_model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=precision
    )
    utils = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
    )
    ssd_model.to(device)  # モデルをデバイスに移動
    ssd_model.eval()
    for image_path in tqdm(image_paths, desc="ssd"):
        try:
            image_filename = os.path.basename(image_path)
            filename, file_extension = os.path.splitext(
                image_filename
            )  # ファイル名と拡張子を分割
            new_filename = f"{filename}_ssd{file_extension}"  # 新しいファイル名を作成

            saved_image_path = os.path.join(
                "{}/data/train/dataSet/images/train".format(full_path),
                new_filename,  # 新しいファイル名を使用
            )

            # 入力データ作成
            inputs = [utils.prepare_input(image_path)]
            tensor = utils.prepare_tensor(inputs, precision == "fp16")
            tensor = tensor.to(device)  # テンソルをデバイスに移動

            # 検知
            with torch.no_grad():
                detections_batch = ssd_model(tensor)

            # クラスのラベル取得
            classes_to_labels = utils.get_coco_object_dictionary()

            # 結果取得
            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [
                utils.pick_best(results, confidence_threshold)
                for results in results_per_input
            ]

            # 描画とデータ保存
            for image_idx in range(len(best_results_per_input)):
                fig, ax = plt.subplots(1)
                image = inputs[image_idx] / 2 + 0.5
                ax.imshow(image)
                bboxes, classes, confidences = best_results_per_input[image_idx]
                for idx in range(len(bboxes)):
                    left, bot, right, top = bboxes[idx]
                    x, y, w, h = [
                        val * 300 for val in [left, bot, right - left, top - bot]
                    ]
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x,
                        y,
                        "{} {:.0f}%".format(
                            classes_to_labels[classes[idx] - 1], confidences[idx] * 100
                        ),
                        bbox=dict(facecolor="white", alpha=0.5),
                    )
                plt.show()

            # データをファイルに保存
            output_file = os.path.join(
                "{}/data/train/dataSet/labels/train".format(full_path),
                os.path.splitext(os.path.basename(image_path))[0] + "_ssd.txt",
            )
            with open(output_file, "w") as file:
                for image_idx in range(len(best_results_per_input)):
                    bboxes, classes, confidences = best_results_per_input[image_idx]
                    try:
                        x1, y1, x2, y2 = bboxes[0]
                    except:
                        os.remove(output_file)
                        break
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    class_number = classes[0]
                    if class_number != 1:
                        os.remove(output_file)
                        break
                    data_string = f"{class_number-1} {cx} {cy} {width} {height}"
                    file.write(data_string + "\n")
                else:
                    shutil.copyfile(image_path, saved_image_path)
                    # print("データをファイルに保存しました:", output_file)
        except:
            os.remove(output_file)
            print("エラーが発生しました")


# 関数の呼び出し
if __name__ == "__main__":
    make_box_text_from_ssd(["./test.jpg"], "../..")
