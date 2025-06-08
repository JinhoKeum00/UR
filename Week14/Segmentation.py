import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# -----------------------------
# 1) Dataset
# -----------------------------
class MNISTSegDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.image_dir = os.path.join(root_dir, split, "image")
        self.gt_dir    = os.path.join(root_dir, split, "gt")
        # 각 파일명에서 "0001_0.png" → id="0001"
        self.ids = sorted({fname.split("_")[0] for fname in os.listdir(self.gt_dir)})

        self.transform_img = T.Compose([
            T.Resize((64,64)),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3)
        ])
        # — 수정: mask는 ToTensor() 하지 않고 raw PIL→np로 읽음
        self.transform_mask = T.Resize((64,64))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # 원본 이미지 로드
        img = Image.open(os.path.join(self.image_dir, f"{img_id}.png")).convert("RGB")
        img = self.transform_img(img)

        masks, cls_labels = [], []
        for i in range(4):
            gp = os.path.join(self.gt_dir, f"{img_id}_{i}.png")
            mask_pil = Image.open(gp)
            mask_resized = self.transform_mask(mask_pil)
            mask_arr = np.array(mask_resized, dtype=np.uint8)  # raw pixel 값 (0 또는 label+1)

            # binary mask
            bin_mask = (mask_arr > 0).astype(np.float32)
            masks.append(torch.from_numpy(bin_mask)[None, ...])  # [1,H,W]

            # class label = pixel_value - 1, 없으면 -1
            vals = np.unique(mask_arr)
            vals = vals[vals > 0]
            cls_labels.append(int(vals[0]) - 1 if len(vals) else -1)

        gt_masks = torch.cat(masks, dim=0)                     # [4,H,W]
        cls_labels = torch.tensor(cls_labels, dtype=torch.long)  # [4]

        return img, gt_masks, cls_labels

# -----------------------------
# 2) DataLoader
# -----------------------------
DATA_ROOT = "/content/mnist_seg_dataset"
train_ds = MNISTSegDataset(DATA_ROOT, split="train")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
test_ds  = MNISTSegDataset(DATA_ROOT, split="test")
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------------
# 3) Model 정의
# -----------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def up_conv(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

# ────────────────────────────────────────────────────────────────
# 2) U-Net + Parallel Class Head
# ────────────────────────────────────────────────────────────────
class UNetSegClassModel(nn.Module):
    def __init__(self, num_instances=4, num_classes=10):
        super().__init__()
        # -- Encoder
        self.enc1 = conv_block(3,  64)
        self.enc2 = conv_block(64,128)
        self.enc3 = conv_block(128,256)
        self.enc4 = conv_block(256,512)
        self.pool = nn.MaxPool2d(2,2)

        # -- Bottleneck
        self.center = conv_block(512,1024)

        # -- Decoder
        self.up4 = up_conv(1024,512)
        self.dec4 = conv_block(1024,512)
        self.up3 = up_conv(512,256)
        self.dec3 = conv_block(512,256)
        self.up2 = up_conv(256,128)
        self.dec2 = conv_block(256,128)
        self.up1 = up_conv(128,64)
        self.dec1 = conv_block(128,64)

        # -- Segmentation Head
        self.seg_head = nn.Conv2d(64, num_instances, kernel_size=1)

        # -- Class Head (bottleneck features → per-instance class)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),               # [B,1024,1,1]
            nn.Conv2d(1024, num_instances*num_classes, 1),
        )
        self.num_instances = num_instances
        self.num_classes   = num_classes

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B,64,H,W]
        p1 = self.pool(e1)          # [B,64,H/2,W/2]
        e2 = self.enc2(p1)          # [B,128,...]
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        c  = self.center(p4)        # [B,1024,H/16,W/16]

        # Decoder
        u4 = self.up4(c)            # [B,512,H/8,W/8]
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Segmentation logits
        seg_logits = self.seg_head(d1)  # [B, num_instances, H, W]

        # Classification logits
        cls_logits = self.cls_head(c)   # [B, num_instances*num_classes, 1,1]
        cls_logits = cls_logits.view(-1, self.num_instances, self.num_classes)

        return seg_logits, cls_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetSegClassModel(num_instances=4, num_classes=10).to(device)

# -----------------------------
# 5) Optimizer / Scheduler / Loss
# -----------------------------
optimizer     = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=1e-5
)
scheduler     = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
seg_criterion = nn.BCEWithLogitsLoss()
cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)

# -----------------------------
# 6) Training Loop
# -----------------------------
def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (2,3)
    inter = torch.sum(probs * targets, dims)
    union = torch.sum(probs + targets, dims)
    return torch.mean((1 - (2*inter + smooth)/(union + smooth)))

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for imgs, gt_masks, cls_targets in train_loader:
        imgs, gt_masks, cls_targets = imgs.to(device), gt_masks.to(device), cls_targets.to(device)

        seg_logits, cls_logits = model(imgs)
        # segmentation loss: BCE + Dice
        loss_bce = seg_criterion(seg_logits, gt_masks)
        loss_dice = dice_loss(seg_logits, gt_masks)
        loss_seg = loss_bce + loss_dice

        # classification loss
        B = cls_logits.size(0)
        cls_flat = cls_logits.view(B*4, -1)
        tgt_flat = cls_targets.view(B*4)
        loss_cls = cls_criterion(cls_flat, tgt_flat)

        loss = loss_seg*3 + loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation loss 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, gt_masks, cls_targets in test_loader:
            imgs, gt_masks, cls_targets = imgs.to(device), gt_masks.to(device), cls_targets.to(device)
            seg_logits, cls_logits = model(imgs)
            # upsample은 model.forward에서 처리済
            loss_bce = seg_criterion(seg_logits, gt_masks)
            loss_dice = dice_loss(seg_logits, gt_masks)
            loss_seg = loss_bce + loss_dice

            B = cls_logits.size(0)
            cls_flat = cls_logits.view(B*4, -1)
            tgt_flat = cls_targets.view(B*4)
            loss_cls = cls_criterion(cls_flat, tgt_flat)

            val_loss += (loss_seg + loss_cls).item()
    val_loss /= len(test_loader)

    scheduler.step(val_loss)  # — 수정: ReduceLROnPlateau

    print(f"[Epoch {epoch+1:02d}] Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

# -----------------------------
# 7) Evaluation & Visualization
# -----------------------------
model.eval()
with torch.no_grad():
    imgs, gt_masks, cls_targets = next(iter(test_loader))
    imgs, gt_masks = imgs.to(device), gt_masks.to(device)
    seg_logits, cls_logits = model(imgs)
    # 이미 model.forward 에 upsample 포함
    seg_prob   = torch.sigmoid(seg_logits).cpu().numpy()  # [B,4,64,64]
    pred_masks = seg_prob > 0.5
    gt_np      = gt_masks.cpu().numpy().astype(bool)
    cls_np     = cls_logits.cpu().numpy()

    # 한 샘플만
    b = 0
    img_t     = imgs[b].cpu()
    gt_b      = gt_np[b]
    pm_b      = pred_masks[b]
    cls_b     = cls_np[b]  # [4,10]

    # Hungarian matching
    N = gt_b.shape[0]
    cost = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            cost[i,j] = -np.sum(gt_b[i] & pm_b[j])
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_masks = pm_b[col_ind]
    matched_cls   = cls_b[col_ind].argmax(axis=1)

    # 이미지 denormalize
    def denorm(t, mean=(0.5,)*3, std=(0.5,)*3):
        t = t.clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c].mul_(s).add_(m)
        return t

    img_np = denorm(img_t).permute(1,2,0).numpy()

# Plot
fig, axes = plt.subplots(2, N+1, figsize=(4*(N+1), 8))
axes[0,0].imshow(img_np); axes[0,0].set_title("Image"); axes[0,0].axis("off")
for i in range(N):
    axes[0,i+1].imshow(gt_b[i], cmap="gray"); axes[0,i+1].set_title(f"GT mask {i}"); axes[0,i+1].axis("off")
axes[1,0].axis("off")
for i in range(N):
    axes[1,i+1].imshow(matched_masks[i], cmap="gray")
    axes[1,i+1].set_title(f"Pred mask {i}\nClass: {matched_cls[i]}")
    axes[1,i+1].axis("off")
plt.tight_layout()
plt.show()
