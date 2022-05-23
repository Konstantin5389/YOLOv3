import torch


def intersection_over_union(boxes1, boxes2, box_format="midpoint"):
    """_summary_

    Args:
        boxes1 (tensor): boxes: [..., 4]
        boxes2 (tensor): should have same shape with boxes1
    """

    if box_format == "midpoint":
        x11 = (boxes1[..., 0] - boxes1[..., 2] / 2).unsqueeze(-1)
        y11 = (boxes1[..., 1] - boxes1[..., 3] / 2).unsqueeze(-1)
        x12 = (boxes1[..., 0] + boxes1[..., 2] / 2).unsqueeze(-1)
        y12 = (boxes1[..., 1] + boxes1[..., 3] / 2).unsqueeze(-1)

        x21 = (boxes2[..., 0] - boxes2[..., 2] / 2).unsqueeze(-1)
        y21 = (boxes2[..., 1] - boxes2[..., 3] / 2).unsqueeze(-1)
        x22 = (boxes2[..., 0] + boxes2[..., 2] / 2).unsqueeze(-1)
        y22 = (boxes2[..., 1] + boxes2[..., 3] / 2).unsqueeze(-1)
    else:
        x11 = boxes1[..., 0:1]
        y11 = boxes1[..., 1:2]
        x12 = boxes1[..., 2:3]
        y12 = boxes1[..., 3:4]

        x21 = boxes2[..., 0]
        y21 = boxes2[..., 1]
        x22 = boxes2[..., 2]
        y22 = boxes2[..., 3]

    x1 = torch.max(x11, x21)
    y1 = torch.max(y11, y21)
    x2 = torch.min(x12, x22)
    y2 = torch.min(y12, y22)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - inter
    return inter / (union + 1e-6)


if __name__ == "__main__":
    boxes1 = torch.tensor([[0, 0, 100, 100]])
    boxes2 = torch.tensor([[0, 0, 200, 200]])
    assert intersection_over_union(boxes1, boxes2) == torch.tensor([0.25])
    print("iou success!")
