import torch
from torch.utils.data import DataLoader
import config
import torch.optim as optim
from model import YOLOv3
from loss import YOLOLoss
from tqdm import tqdm
from dataset import VOCDataset
import pdb
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train(model, train_loader, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        losses = []
        x = x.to(config.DEVICE)
        y[0] = y[0].to(config.DEVICE)
        y[1] = y[1].to(config.DEVICE)
        y[2] = y[2].to(config.DEVICE)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y[0], scaled_anchors[0]) +
                loss_fn(out[1], y[1], scaled_anchors[1]) +
                loss_fn(out[2], y[2], scaled_anchors[2])
            )
        optimizer.zero_grad()
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=(sum(losses) / len(losses)))


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_set = VOCDataset(root_dir="data/", anchors=config.ANCHORS, phase="train", feature_size=config.S,
                           image_size=config.IMAGE_SIZE, transform=config.train_transforms, num_classes=config.NUM_CLASSES)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_set = VOCDataset(root_dir="data/", anchors=config.ANCHORS, phase="test", feature_size=config.S,
                          image_size=config.IMAGE_SIZE, transform=config.test_transforms, num_classes=config.NUM_CLASSES)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    scaled_anchors = (torch.tensor(
        config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train(model, train_loader, optimizer, loss_fn, scaler, scaled_anchors)


if __name__ == "__main__":
    main()
