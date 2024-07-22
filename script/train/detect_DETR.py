import os
import warnings

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet50
from tqdm import tqdm

# UserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning)


def save_results(
    a, image, image_path, boxes, prob, normalized_boxes, classes, full_path
):
    # 元の画像の幅と高さを取得
    original_width, original_height = image.size

    image_filename = os.path.basename(image_path)
    filename, file_extension = os.path.splitext(image_filename)
    new_filename = f"{filename}_DETR{file_extension}"

    # 元の画像を保存
    saved_image_path = os.path.join(
        "{}/data/train/dataSet/images/train".format(full_path),
        new_filename,
    )

    a.save(saved_image_path)

    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    colors = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ] * 100
    tmp = []
    for p, (xmin, ymin, xmax, ymax), norm_box, c in zip(
        prob, boxes.tolist(), normalized_boxes.tolist(), colors
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cx_norm = (norm_box[0] + norm_box[2]) / 2
        cy_norm = (norm_box[1] + norm_box[3]) / 2
        cl = p.argmax()
        seiki_width = (xmax - xmin) / original_width
        seiki_height = (ymax - ymin) / original_height
        text = f" {0} {p[cl]:0.2f} {cx_norm} {cy_norm} {seiki_width} {seiki_height}"
        tmp.append([p[cl], cx_norm, cy_norm, seiki_width, seiki_height])
        ax.text(
            xmin, ymin - 10, text, fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5)
        )
        # print(f"Normalized Box: {text}")
    tmp.sort(key=lambda x: x[0], reverse=True)
    string = f"{0} {tmp[0][1]} {tmp[0][2]} {tmp[0][3]} {tmp[0][4]}"
    # print(string)
    output_file = os.path.join(
        "{}/data/train/dataSet/labels/train".format(full_path),
        os.path.splitext(os.path.basename(image_path))[0] + "_DETR.txt",
    )
    plt.close()
    with open(output_file, "a") as file:
        file.write(string + "\n")


def make_box_text_from_DETR(paths, full_path):
    class DETRdemo(nn.Module):
        def __init__(
            self,
            num_classes,
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
        ):
            super().__init__()
            self.backbone = resnet50()
            del self.backbone.fc
            self.conv = nn.Conv2d(2048, hidden_dim, 1)
            self.transformer = nn.Transformer(
                hidden_dim, nheads, num_encoder_layers, num_decoder_layers
            )
            self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
            self.linear_bbox = nn.Linear(hidden_dim, 4)
            self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        def forward(self, inputs):
            x = self.backbone.conv1(inputs)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            h = self.conv(x)
            H, W = h.shape[-2:]
            pos = (
                torch.cat(
                    [
                        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                    ],
                    dim=-1,
                )
                .flatten(0, 1)
                .unsqueeze(1)
            )
            h = self.transformer(
                pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
            ).transpose(0, 1)
            return {
                "pred_logits": self.linear_class(h),
                "pred_boxes": self.linear_bbox(h).sigmoid(),
            }

    model = DETRdemo(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
        map_location="cpu",
        check_hash=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    CLASSES = [
        "N/A",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    for path in tqdm(paths, desc="DETR"):
        try:
            image_path = path
            a = Image.open(image_path)
            im = Image.open(image_path)

            transform = transforms.Compose(
                [
                    transforms.Resize((800, 1333)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            img = transform(im).unsqueeze(0)
            outputs = model(img)

            def box_cxcywh_to_xyxy(x):
                x_c, y_c, w, h = x.unbind(1)
                b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
                return torch.stack(b, dim=1)

            def rescale_bboxes(out_bbox, size):
                img_w, img_h = size
                b = box_cxcywh_to_xyxy(out_bbox)
                b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                return b

            threshold = 0.7
            probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold
            bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)
            prob = probas[keep]

            if "person" in [CLASSES[p.argmax()] for p in prob]:
                save_results(
                    a,
                    im,
                    image_path,
                    bboxes_scaled,
                    prob,
                    outputs["pred_boxes"][0, keep],
                    CLASSES,
                    full_path,
                )

        except Exception as e:
            print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    paths = ["test.jpg"]  # パスのリストを入力する
    make_box_text_from_DETR(paths, "../..")
