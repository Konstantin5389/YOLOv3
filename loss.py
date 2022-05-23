import torch
import torch.nn as nn
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, ) -> None:
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.no_obj = 10
        self.box = 10

    def forward(self, predictions, targets, scaled_anchors):
        """_summary_

        Args:
            predictions (tensor): (B, Anchors, H, W, C + 5)
            targets (tensor): (B, Anchors, H, W, 6) targets[..., :] = [obj, x_off, y_off, w, h, label]
            scaled_anchors (tensor): (3, 2)
        """
        obj = targets[..., 0] == 1
        no_obj = targets[..., 0] == 0
        scaled_anchors = scaled_anchors.reshape(1, 3, 1, 1, 2)

        # -----no obj loss------ #
        no_obj_loss = self.bce(
            predictions[..., 0:1][no_obj],
            targets[..., 0:1][no_obj]
        )

        # -----obj loss------ #
        pred_boxes = torch.cat((self.sigmoid(predictions[..., 1:3]), torch.exp(
            predictions[..., 3:5]) * scaled_anchors), dim=-1)
        ious = intersection_over_union(pred_boxes, targets[..., 1:5])
        obj_loss = self.mse(
            self.sigmoid(predictions[..., 0][obj]),
            ious * targets[..., 0][obj]
        )

        # -----bbox loss------ #
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        targets[..., 3:5] = torch.log(
            targets[..., 3:5] / scaled_anchors + 1e-16)
        box_loss = self.mse(
            predictions[..., 1:5],
            targets[..., 1:5]
        )

        # -----class loss------ #
        class_loss = self.entropy(
            predictions[..., 5:][obj],
            targets[..., 5][obj].long()
        )

        return self.no_obj * no_obj_loss + obj_loss + self.box * box_loss + class_loss


if __name__ == "__main__":
    target = torch.randn(2, 3, 13, 13, 6)
    target[..., 0] = 0
    target[..., 0::2, 1::2, 0] = 1
    target[..., 3:5] = torch.exp(target[..., 3:5])
    target[..., 5:6] = torch.randint(0, 19, (2, 3, 13, 13, 1))
    predictions = torch.randn((2, 3, 13, 13, 25))
    criterion = YOLOLoss()
    scaled_anchors = torch.tensor(
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]) * 13
    print(criterion(predictions, target, scaled_anchors))
