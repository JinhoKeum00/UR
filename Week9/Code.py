import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MNISTSegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform_img=None, transform_mask=None):

        # root_dir (str) -> 데이터셋 경로 -> "/content/mnist_seg_dataset"
        # split (str) -> "train" 또는 "test" 입력
        # transform_img: 이미지 전처리용 transforms.Compose
        # transform_mask: 마스크 전처리용 transforms.Compose

        # 디렉토리를 image, gt가 담긴 곳으로 지정
        self.image_dir = os.path.join(root_dir, split, "image")
        self.gt_dir    = os.path.join(root_dir, split, "gt")

        # 확장자를 제거한 파일 목록 만들기
        self.ids = sorted([fname.split(".")[0] for fname in os.listdir(self.image_dir)])

        # 이미지 및 마스크 전처리
        self.transform_img  = transform_img or T.Compose([
            T.ToTensor(),                     # [0,255] -> [0,1]
            T.Normalize((0.5,)*3, (0.5,)*3)   # [0,1] -> [-1,1]
        ])
        self.transform_mask = transform_mask or T.Compose([
            T.ToTensor()                      # 1×H×W -> 값은 0 혹은 1
        ])

    # 데이터셋 전체 샘플 개수 반환
    def __len__(self):
        return len(self.ids)

    # idx번째 이미지 및 GT 반환
    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 1) 원본 이미지 로드
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB") # 3채널 컬러
        image = self.transform_img(image)  # 3×64×64 Tensor 형태로 변환

        # 2) 4개의 GT 마스크 로드
        masks = []
        for i in range(4):
            mask_path = os.path.join(self.gt_dir, f"{img_id}_{i}.png")
            mask = Image.open(mask_path).convert("L")  # 1채널 흑백
            mask = self.transform_mask(mask) # Tensor: 1×64×64
            masks.append(mask)

        # 3) 4개 GT를 하나로 표현
        # [1×64×64] -> [4×64×64] 텐서로 결합
        gt_masks = torch.cat(masks, dim=0) # 채널 축으로 이어 붙임

        return image, gt_masks
    
# Colab 환경에서 루트 폴더 경로
DATA_ROOT = "/content/mnist_seg_dataset"

# Dataset, DataLoader 준비
# 학습
# 이미지 -> [3,64,64] / GT -> [4,64,64]
train_ds = MNISTSegDataset(
    root_dir=DATA_ROOT,
    split="train",
    transform_img=T.Compose([
        # 스케일링, 정규화
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3)
    ]),
    transform_mask=T.Compose([
        # 스케일링
        T.Resize((64,64)),
        T.ToTensor()
    ])
)
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=2, # 데이터 background 전처리
    pin_memory=True # GPU 최적화
)

# 테스트
test_ds = MNISTSegDataset(root_dir=DATA_ROOT, split="test")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# 데이터 상태 한번 확인
for imgs, gts in train_loader:
    print(imgs.shape)  # [16, 3, 64, 64] -> 배치사이즈 16
    print(gts.shape)   # [16, 4, 64, 64] -> 배치사이즈 16
    break

import matplotlib.pyplot as plt

# 데이터에서 배치 사이즈만큼 데이터 불러오기
imgs, gts = next(iter(train_loader))  # imgs: [16,3,64,64], gts: [16,4,64,64]

# 배치에서 첫 샘플 꺼내기
img = imgs[0]    # [3,64,64] -> 이미지 한장
masks = gts[0]   # [4,64,64] -> GT 4장

# 정규화된 이미지 기존 스케일로 복원
# -> 정규화된 상태에서는 이미지가 잘 보이지 않기 때문
def denormalize(tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# 원본 텐서 보존 및 복사된 텐서에 denormalize 적용
# permute로 C×H×W -> H×W×C로 변환 => NumPy 및 Matplotlib 호환 가능
img = denormalize(img.clone()).permute(1,2,0).cpu().numpy()  # H×W×C, [0~1]

# Plot -> 원본 이미지 + GT 4장 가로 방향으로 한 번에 배열
fig, axes = plt.subplots(1, 5, figsize=(15,4))

# 첫 번째 칸 -> 기존 이미지
axes[0].imshow(img)
axes[0].set_title("Image")
axes[0].axis("off")

# 2~5번째 칸 -> GT 4장
for i in range(4):
    mask = masks[i].cpu().numpy()  # [64,64], 0 또는 1
    axes[i+1].imshow(mask, cmap="gray")
    axes[i+1].set_title(f"GT mask {i}")
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()

# Backbone(ResNet50) + Decoder + Segmentation Head 구조
from torchvision.models.segmentation import fcn_resnet50

# ImageNet으로 사전 학습된 ResNet50 가중치
from torchvision.models import ResNet50_Weights
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 선언: backbone만 ImageNet pre-train, head은 학습 필요
model = fcn_resnet50(
    # segmentation head는 pre-trained weight X
    weight=None,
    # Backbone은 pre-trained weight O
    weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
    # 4개의 GT mask 채널
    num_classes=4
).to(device)

# Backbone은 학습 과정에서 변경 X
for param in model.backbone.parameters():
    param.requires_grad = False

# optimizer -> requires_grad == True를 기준으로 Head 파라미터만 학습
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001, momentum=0.9, weight_decay=0.0005
)

# 멀티채널 바이너리 로스
criterion = nn.BCEWithLogitsLoss()

# 따라서, backbone을 활용하여 새로운 데이터에 맞춘 Head만 학습한다.(Fine-Tuning)

num_epochs = 30

for epoch in range(num_epochs):
  model.train()
  cost = 0.0

  for images, targets in train_loader:
    images = images.to(device) # [16x3x64x64] -> 이미지 한장
    targets = targets.to(device) # [16x4x64x64] -> GT 4장

    # FCN-ResNet50 모델은 {"out": tensor, "aux": tensor}와 같은 딕셔너리 형태
    # 실제 손실 함수에 사용할 형태 직접 대입
    output = model(images)["out"]

    loss = criterion(output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost += loss
  cost = cost / len(train_loader)
  print(f"Epoch : {epoch+1:4d}, Cost : {cost:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

model.eval()
with torch.no_grad():
    imgs, gt_masks = next(iter(test_loader))
    imgs, gt_masks = imgs.to(device), gt_masks.to(device)

    # 예측
    out = model(imgs)['out']
    pred_full = out.argmax(dim=1).cpu().numpy()  # [B,H,W]

    # 기본은 0번째 샘플로 되어있는데, 다른 샘플은 index를 수정하여 확인할 수 있습니다.
    img   = imgs[0].cpu()                        # [3,H,W]
    gt    = gt_masks[0].cpu().numpy()            # [4,H,W]
    pred  = pred_full[0]                         # [H,W]

    # denormalize + permute
    img = img.mul(0.5).add(0.5).permute(1,2,0).numpy()

    n_ch    = gt.shape[0]       # 4
    num_cls = out.shape[1]      # 11 (0~9 + None=10)
    cost    = np.zeros((n_ch, num_cls), dtype=np.int32)

    # 비용 행렬 만들기 (mask가 비어있는 채널은 모두 0으로)
    for i in range(n_ch):
        mask_i = gt[i].astype(bool)
        if not mask_i.any():
            # GT 없는 채널이면 cost row는 그대로 0
            continue
        for j in range(num_cls):
            cost[i,j] = -np.sum((pred == j) & mask_i)

    # 헝가리안 알고리즘으로 매칭
    row_ind, col_ind = linear_sum_assignment(cost)
    # 이제 row_ind[k] 채널 ↔ col_ind[k] 클래스 매핑

    # 각 채널별 assigned class 정리 (없으면 -1)
    assigned = {i: -1 for i in range(n_ch)}
    for i, j in zip(row_ind, col_ind):
        # 매칭됐더라도 GT가 비어 있었던 채널(i)은 제외
        if gt[i].astype(bool).any():
            assigned[i] = j

    # 시각화
    fig, axes = plt.subplots(n_ch, 3, figsize=(9, 3*n_ch))
    if n_ch == 1: axes = axes[np.newaxis, :]

    for ch in range(n_ch):
        mask_ch      = gt[ch].astype(bool)
        cls_pred     = assigned[ch] # 0~10 혹은 -1
        # GT영역 & 예측 클래스가 같을 때만 표시
        if cls_pred >= 0:
            pm = ((pred == cls_pred) & mask_ch).astype(float)
        else:
            pm = np.zeros_like(pred, dtype=float)
        pm = np.clip(pm*2.0, 0, 1) # 선명도 2배

        # GT 시각화
        gt_viz = mask_ch.astype(float)

        # plot 원본
        ax = axes[ch, 0]
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis("off")

        # plot GT mask
        ax = axes[ch, 1]
        ax.imshow(gt_viz, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"GT ch {ch}")
        ax.axis("off")

        # plot Predicted mask
        ax = axes[ch, 2]
        ax.imshow(pm, cmap="gray", vmin=0, vmax=1)
        # 클래스가 -1 이면 “None”으로, 아니면 숫자로 표시
        title = "Pred cls None" if cls_pred < 0 else f"Pred cls {cls_pred}"
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
