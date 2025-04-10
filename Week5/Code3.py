import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

# 1) CustomDataset 구성 (MNIST 데이터셋)
class CustomMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in range(10):
            label_dir = os.path.join(root_dir, str(label))
            for img_file in glob.glob(os.path.join(label_dir, '*.png')):
                self.image_paths.append(img_file)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("L")  # 흑백 이미지로 변환
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


# 2) Encoder와 Decoder 클래스 (이전 학습한 모델)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # flatten
        return out

# Decoder는 여기서는 사용하지 않으므로 생략 가능

# 3) Head 클래스 정의 (Classifier)
class Head(nn.Module):
    def __init__(self, input_dim=256*7*7, num_classes=10):
        super(Head, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

# 4) Head 학습을 위한 함수 (encoder는 고정된 상태)
def train_head(encoder, head, dataloader, optimizer, loss_func, device):
    head.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # encoder는 eval 모드이므로, with torch.no_grad()를 사용하여 feature 추출
        with torch.no_grad():
            features = encoder(images)

        # head를 통한 분류 예측
        outputs = head(features)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# 5) 학습 준비 및 진행
if __name__ == '__main__':
    # 설정값
    root_dir = './data/train'  # MNIST 데이터가 저장된 디렉토리 (각 폴더 이름은 0~9)
    batch_size = 256
    num_epochs = 5     # Head만 학습할 경우 일반적으로 적은 epoch로도 가능
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 데이터셋 및 DataLoader 구성
    dataset = CustomMNISTDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Encoder 인스턴스 생성 및 저장된 가중치 불러오기
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("encoder_weights.pth"))
    encoder.eval()  # evaluation 모드로 변경
    # encoder의 가중치 업데이트를 막음
    for param in encoder.parameters():
        param.requires_grad = False

    # Head 인스턴스 생성 (분류기)
    head = Head().to(device)

    # Head 학습을 위한 옵티마이저와 손실 함수 정의 (CrossEntropyLoss는 분류 문제에 주로 사용됨)
    optimizer_head = optim.Adam(head.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # Head 학습 루프
    for epoch in range(num_epochs):
        avg_loss, accuracy = train_head(encoder, head, dataloader, optimizer_head, loss_func, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 학습 완료 후, Head 모델 저장 (원한다면)
    torch.save(head.state_dict(), "head_weights.pth")
    print("Head weights saved to head_weights.pth")

def evaluate(encoder, head, dataloader, device):
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = encoder(images)
            outputs = head(outputs)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for i in range(len(images)):
                sample_count += 1

                if sample_count % 100 == 0:
                    actual = labels[i].item()
                    predicted = predictions[i].item()
                    probs = outputs[i].cpu().numpy().round(3)
                    print(f"[Sample {sample_count}] 실제: {actual}, 예측: {predicted}, 0~9 확률: {probs}")
                    print("-" * 40)

    accuracy = correct / total * 100
    print(f"\n 테스트 정확도: {accuracy:.2f}% ({correct}/{total})")

# 테스트 데이터 로드
test_dataset = CustomMNISTDataset(root_dir="./data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 평가 실행
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("encoder_weights.pth"))
head = Head().to(device)
head.load_state_dict(torch.load("head_weights.pth"))
evaluate(encoder, head, test_loader, device)