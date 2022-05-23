import torch
import numpy as np
import PIL.Image as Image
import os
import pandas as pd
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root_dir, anchors, phase="train", feature_size=[13, 26, 52], image_size=416,
                 transform=None, num_classes=20) -> None:
        super(VOCDataset, self).__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.feature_size = feature_size
        self.image_size = 416
        self.C = num_classes
        self.transform = transform
        self.list = pd.read_csv(os.path.join(root_dir, phase + ".csv"))
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_all_anchors = len(self.anchors)
        self.A = self.num_all_anchors // len(feature_size)
        self.iou_thresh = 0.5

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image = np.array(Image.open(os.path.join(
            self.root_dir, "images/", self.list.iloc[index, 0])).convert("RGB"))
        boxes = np.roll(np.loadtxt(os.path.join(self.root_dir, "labels/",
                        self.list.iloc[index, 1]), delimiter=" ", ndmin=2), 4, axis=1).tolist()

        if self.transform:
            argumentations = self.transform(image=image, bboxes=boxes)
            image, boxes = argumentations['image'], argumentations['bboxes']

        targets = [torch.zeros((self.A, S, S, 6)) for S in self.feature_size]
        for box in boxes:
            flag = False
            anchor_ious = self.iou_wh(torch.tensor(box[2:4]), self.anchors)
            anchor_ids = torch.argsort(anchor_ious, descending=True)
            for anchor_id in anchor_ids:
                feature_idx = anchor_id // self.A
                anchor_idx = anchor_id % self.A
                x_cell = int(box[0] * self.feature_size[feature_idx])
                y_cell = int(box[1] * self.feature_size[feature_idx])
                x_off = box[0] * self.feature_size[feature_idx] - x_cell
                y_off = box[1] * self.feature_size[feature_idx] - y_cell
                # targets[feature_idx][anchor_idx, y_cell, x_cell, :]
                if targets[feature_idx][anchor_idx, y_cell, x_cell, 0] == 0 and flag == False:
                    _, _, w, h, label = box
                    w_cell = w * self.feature_size[feature_idx]
                    h_cell = h * self.feature_size[feature_idx]
                    targets[feature_idx][anchor_idx, y_cell, x_cell, 0] = 1
                    targets[feature_idx][anchor_idx, y_cell, x_cell,
                                         1:5] = torch.tensor([x_off, y_off, w_cell, h_cell])
                    targets[feature_idx][anchor_idx,
                                         y_cell, x_cell, 5] = int(label)
                    flag = True
                elif targets[feature_idx][anchor_idx, y_cell, x_cell, 0] == 0 and anchor_ious[anchor_id] > self.iou_thresh:
                    targets[feature_idx][anchor_idx, y_cell, x_cell, 0] = -1
        return image, targets

    def iou_wh(self, box, boxes):
        """_summary_

        Args:
            box (tensor): the ground turth boxes
            boxes (tensor): the anchors
        Format:
            mid point
        """
        inter = torch.min(box[..., 0], boxes[..., 0]) * \
            torch.min(box[..., 1], boxes[..., 1])
        union = box[..., 0] * box[..., 1] + \
            boxes[..., 0] * boxes[..., 1] - inter
        return inter / (union + 1e-6)


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2

    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
    IMAGE_SIZE = 416
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="yolo", min_visibility=0.4, label_fields=[]),
    )

    dataset = VOCDataset(root_dir='/home/ljy/YOLOv3/data/', anchors=ANCHORS,
                         phase='test', transform=test_transforms)
    image, targets = dataset[0]
    targets = (targets[0], targets[1], targets[2])
    image_cv2 = image.mul_(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    for target in targets:
        a, h, w, channel = target.shape
        target = target.reshape(a, -1, channel)
        boxes = []
        for i in range(a):
            for j in range(h * w):
                box = target[i, j, :]
                if box[0] != 1:
                    continue
                else:
                    x_cell = j % w
                    y_cell = j // w
                    w_cell = box[3]
                    h_cell = box[4]
                    x = x_cell + box[1]
                    y = y_cell + box[2]
                    x = int(x / w * IMAGE_SIZE)
                    y = int(y / h * IMAGE_SIZE)
                    w_int = int(w_cell / w * IMAGE_SIZE)
                    h_int = int(h_cell / h * IMAGE_SIZE)
                    x1 = x - w_int // 2
                    x2 = x + w_int // 2
                    y1 = y - h_int // 2
                    y2 = y + h_int // 2
                    boxes.append([x1, y1, x2, y2])
        for box in boxes:
            image = cv2.rectangle(
                image_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
    cv2.imwrite('v3test.jpg', image)
