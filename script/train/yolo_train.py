import warnings

from ultralytics import YOLO

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)


def yolo8_train(full_path, new_train=True):
    if new_train:
        model = YOLO("{}/yolov8n.pt".format(full_path))  # create a new model
    else:
        model = YOLO(
            "{}/yolov8_train.pt".format(full_path)
        )  # load a pretrained model (recommended for training)
        # Use the model
    results = model.train(
        data="{}/script/train/dataset.yaml".format(full_path),
        exist_ok=True,
        epochs=5,
        batch=50,
        name="trainYolov8n",
        device="cuda",
    )  # train the model


if __name__ == "__main__":
    yolo8_train("../..", new_train=True)
