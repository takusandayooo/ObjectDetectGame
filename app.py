import base64
import os
import time

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory, url_for
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

model = None
model_train = None

full_path = os.path.abspath(os.curdir)


# NOTE: 元の写真を引数として受け取り、バウンディングボックスを描画し、写真を保存する関数
def rial_time_make_box_and_photo(img, error_count):
    global count, model_number
    im1 = img
    im1_cv = np.array(im1)
    im1_cv = im1_cv[:, :, ::-1].copy()

    boolean = False
    items_train = []
    # NOTE: 学習済みモデルを使用する場合
    if model_number == 2:
        results_train = model_train.predict(source=im1)
        items_train = results_train[0]
        cls_train = -1
        score_train = 0
        label_train = ""
        x1_train = -1
        y1_train = -1
        x2_train = -1
        y2_train = -1

        cls_normal = -1
        score_normal = 0
        label_normal = ""
        x1_normal = -1
        y1_normal = -1
        x2_normal = -1
        y2_normal = -1

        for item in items_train:
            cls = int(item.boxes.cls)  # cls, (N, 1)
            cls_train = int(item.boxes.cls)
            label_train = item.names[int(cls)]
            score_train = item.boxes.conf.to("cuda").cpu().numpy()[0]
            x1_train, y1_train, x2_train, y2_train = (
                item.boxes.xyxy.to("cuda").cpu().numpy()[0]
            )
            print(
                "train",
                cls_train,
                score_train,
                label_train,
                x1_train,
                y1_train,
                x2_train,
                y2_train,
            )
        results = model.predict(source=im1)  # save plotted images
        items_normal = results[0]
        for item in items_normal:
            cls = int(item.boxes.cls)  # cls, (N, 1)
            cls_normal = int(item.boxes.cls)  # cls, (N, 1)
            label_normal = item.names[int(cls)]
            score_normal = (
                item.boxes.conf.to("cuda").cpu().numpy()[0]
            )  # confidence score, (N, 1)
            x1_normal, y1_normal, x2_normal, y2_normal = (
                item.boxes.xyxy.to("cuda").cpu().numpy()[0]
            )  # box with xyxy format, (N, 4)
            print(
                "normal",
                cls_normal,
                score_normal,
                label_normal,
                x1_normal,
                y1_normal,
                x2_normal,
                y2_normal,
            )
        if cls_train == 0 and 0.7 <= score_train:
            boolean = False
            if error_count == 0:
                cv2.rectangle(
                    im1_cv,
                    (int(x1_train), int(y1_train)),
                    (int(x2_train), int(y2_train)),
                    (0, 255, 0),
                    2,
                )
            else:
                # NOTE: error_countが0でない場合は、赤色の枠を表示
                cv2.rectangle(
                    im1_cv,
                    (int(x1_train), int(y1_train)),
                    (int(x2_train), int(y2_train)),
                    (0, 0, 255),
                    2,
                )
            return {
                "boolean": boolean,
                "img": im1_cv,
                "label": label_normal,
                "score": score_train,
            }
        else:
            if cls_normal != 0 or 0.2 <= score_normal <= 0.7:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Add this line
                cv2.imwrite(
                    "{}/data/train/photos/".format(full_path)
                    + str(round(time.time() * 1000))
                    + ".jpg",
                    img,
                )
                count += 1
                boolean = True
                cv2.rectangle(
                    im1_cv,
                    (int(x1_normal), int(y1_normal)),
                    (int(x2_normal), int(y2_normal)),
                    (0, 255, 0),
                    2,
                )
            elif score_normal < 0.2:
                pass
            else:
                boolean = False
                if error_count == 0:
                    cv2.rectangle(
                        im1_cv,
                        (int(x1_normal), int(y1_normal)),
                        (int(x2_normal), int(y2_normal)),
                        (0, 255, 0),
                        2,
                    )
                else:
                    # NOTE: error_countが0でない場合は、赤色の枠を表示
                    cv2.rectangle(
                        im1_cv,
                        (int(x1_normal), int(y1_normal)),
                        (int(x2_normal), int(y2_normal)),
                        (0, 0, 255),
                        2,
                    )
            return {
                "boolean": boolean,
                "img": im1_cv,
                "label": label_normal,
                "score": score_normal,
            }

    # NOTE: 学習済みモデルを使用しない場合
    if len(items_train) == 0:
        results = model.predict(source=im1)  # save plotted images
        items = results[0]
        for item in items:
            cls = int(item.boxes.cls)  # cls, (N, 1)
            label = item.names[int(cls)]
            score = (
                item.boxes.conf.to("cuda").cpu().numpy()[0]
            )  # confidence score, (N, 1)
            x1, y1, x2, y2 = (
                item.boxes.xyxy.to("cuda").cpu().numpy()[0]
            )  # box with xyxy format, (N, 4)
            print(cls, score, label, x1, y1, x2, y2)

            if cls != 0 or 0.2 <= score <= 0.7:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Add this line
                cv2.imwrite(
                    "{}/data/train/photos/".format(full_path)
                    + str(round(time.time() * 1000))
                    + ".jpg",
                    img,
                )
                count += 1
                boolean = True
                cv2.rectangle(
                    im1_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            elif score < 0.2:
                pass
            else:
                boolean = False
                if error_count == 0:

                    cv2.rectangle(
                        im1_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                else:
                    # NOTE: error_countが0でない場合は、赤色の枠を表示
                    cv2.rectangle(
                        im1_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                    )
            return {"boolean": boolean, "img": im1_cv, "label": label, "score": score}
    if len(items) == 0:
        return {"boolean": True, "img": im1_cv, "label": "None", "score": 0}


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
model_number = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/select_model", methods=["GET"])
def select_model():
    global model_number, model, model_train
    selected_model = request.args.get("model", "")
    if selected_model == "second":
        model_number = 2
        print(torch.cuda.memory_allocated())
        model = YOLO("./yolov8n.pt")
        model = model.to("cuda")
        print(torch.cuda.memory_allocated())

        model_train = YOLO("./yolov8_train.pt")
        model_train = model_train.to("cuda")
        print(torch.cuda.memory_allocated())
    else:
        model_number = 0
        model = YOLO("./yolov8n.pt")
        model = model.to("cuda")

    # モデルの番号をコンソールに出力
    print("選択されたモデルの番号:", model_number)

    return render_template("select_model.html")


@app.route("/start_count")
def start_count():
    # NOTE: カウントの初期化をおこなわないと、リスタートの際にカウントが残ってしまう
    global count, flame, error_count
    count = 0
    flame = 0
    error_count = 0
    print("start_count", count, flame, error_count)
    return render_template("start_count.html")


@app.route("/game")
def game():
    return render_template("game.html")


@app.route("/gameover")
def gameover():
    return render_template("gameover.html")


@app.route("/gameclear")
def gameclear():
    return render_template("gameclear.html")


@app.route("/music/<path:filename>")
def play(filename):
    return send_from_directory("music", filename)


flame = 0
error_count = 0


@socketio.on("image")
def handle_image(image):
    global count, error_count
    # 画像データをデコード
    image_data = base64.b64decode(image.split(",")[1])
    # OpenCVで画像を読み込む
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Add this line
    dicts = rial_time_make_box_and_photo(img, error_count)
    boolean = dicts["boolean"]
    bounding_box_img = dicts["img"]
    label = dicts["label"]
    score = dicts["score"]

    if boolean == False:
        error_count += 1
    else:
        error_count = 0
    if error_count >= 5:
        print("終了")
        emit(
            "redirect", {"url": url_for("gameover")}
        )  # クライアントにリダイレクトを指示
        return  # この場合、処理を終了します
    emit(
        "label_and_score",
        {"label": label, "score": float(score), "error_count": error_count},
    )
    count += 1
    # 反転した画像をエンコードしてクライアントに送信
    _, buffer = cv2.imencode(".jpg", bounding_box_img)
    inverted_image_data = base64.b64encode(buffer)
    inverted_image_str = "data:image/jpeg;base64," + inverted_image_data.decode("utf-8")
    emit("processed_image", inverted_image_str)


if __name__ == "__main__":
    # TODO: 最終的には、以下のコードをコメントアウトする
    path = os.getcwd().split("/")
    if path[-1] != "main":
        os.chdir("main")
        print("カレントディレクトリを変更しました")
    # NOTE: カウントの初期化
    count = 0
    socketio.run(app, debug=True)
