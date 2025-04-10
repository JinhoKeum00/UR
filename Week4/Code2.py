import os

import glob

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from PIL import Image

import matplotlib.pyplot as plt

import torchvision.utils as vutils

import torch.nn.utils as nn_utils


class CustomMNISTDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.image_paths = []

        self.labels = [] # 레이블을 저장할 리스트 - 추가


        for label in range(10):

            label_dir = os.path.join(root_dir, str(label))

            for img_file in glob.glob(os.path.join(label_dir, '*.png')):

                self.image_paths.append(img_file)
                self.labels.append(label) # 레이블 추가



        print(f'Found {len(self.image_paths)} images')



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, index):

        img_path = self.image_paths[index]

        image = Image.open(img_path).convert("L")

        label = self.labels[index] # 레이블 가져오기 - 추가

        if self.transform is not None:

            image = self.transform(image)



        return image, label # 레이블 반환 - 추가
#--------------------------------------- 여기까지 완료
#--------------------------------------- 데이터, 레이블 불러오기

class MNISTClassifier(nn.Module):
    def __init__(self, img_channels=1, feature_d=64, num_classes=10):
        super(MNISTClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, kernel_size=4, stride=2, padding=1, bias=False),  # 28x28 -> 14x14
            nn.BatchNorm2d(feature_d),
            nn.LeakyReLU(True),

            nn.Conv2d(feature_d, feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14 -> 7x7
            nn.BatchNorm2d(feature_d * 4),
            nn.ReLU(True),

            nn.Conv2d(feature_d * 4, 128, kernel_size=3, stride=2, padding=0, bias=False),  # 7x7 -> 3x3
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                      # (B, 128, 3, 3) -> (B, 1152)
            nn.Linear(128 * 3 * 3, num_classes)  # Fully connected layer
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x  # 출력: (Batch, 10) => Softmax는 손실 함수에서 적용


data_dir = './data/train/'

result_dir = './result/'

os.makedirs(result_dir, exist_ok=True)



# 4. 하이퍼파라미터 정의

batch_size = 64

lr = 0.0001

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 100  # 훈련 에포크 수



print(f"Using Device: {device}")

# 5. 데이터 변환 정의

transform = transforms.Compose([

    transforms.Resize((16, 16)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])



# 6. 데이터셋 및 데이터로더 생성

train_dataset = CustomMNISTDataset(root_dir=data_dir, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# 7. 모델 초기화

disc = Discriminator(img_channels=1, feature_d=32).to(device)





# 8. 최적화 함수 및 손실 함수 정의

opt_gen = optim.Adam(gen.parameters(), lr=lr)

opt_disc = optim.Adam(disc.parameters(), lr=lr)

bce_loss = nn.BCELoss()

smooth_loss = nn.SmoothL1Loss()



# 9. 생성된 이미지 시각화 함수 정의

def show_generated_images(generator, num_images=64, epoch=0):

    generator.eval()

    with torch.no_grad():

        noise = torch.randn(num_images, z_dim, 1, 1).to(device)  # CNN은 4D 텐서 입력을 기대

        fake_images = generator(noise)

        fake_images = fake_images.cpu()

        grid = vutils.make_grid(fake_images, nrow=8, normalize=True)

        plt.clf()

        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')

        plt.axis('off')

        plt.title(f'Generated Images at Epoch {epoch}')

        plt.show()

        plt.savefig(f'{result_dir}image.png', pad_inches=0.1, bbox_inches='tight')

        plt.close()

    generator.train()



# 10. GAN 훈련 루프

for epoch in range(epochs):

    for batch_idx, real in enumerate(train_dataloader):

        real = real.to(device)

        batch_size = real.shape[0]



        ### (a) 판별자 훈련 (진짜 이미지)

        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

        fake = gen(noise)



        # 진짜 이미지 판별

        disc_real = disc(real).view(-1)

        loss_disc_real = bce_loss(disc_real, torch.ones_like(disc_real))



        # 가짜 이미지 판별

        disc_fake = disc(fake.detach()).view(-1)

        loss_disc_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))



        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        opt_disc.zero_grad()

        loss_disc.backward()

        opt_disc.step()



        ### (b) 생성자 훈련 (가짜 이미지)

        output = disc(fake).view(-1)

        loss_gen = bce_loss(output, torch.ones_like(output))



        opt_gen.zero_grad()

        loss_gen.backward()

        opt_gen.step()



        if batch_idx % 100 == 0:

            print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_dataloader)} "

                  f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")



            show_generated_images(gen, num_images=16, epoch=epoch+1)