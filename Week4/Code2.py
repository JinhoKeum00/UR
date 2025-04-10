import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

# 1) CustomDataset 구성
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
        image = Image.open(img_path).convert("L")  # 흑백 이미지
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


# 2) 판별자 클래스 (Encoder + Classifier)
class MNISTClassifier(nn.Module):
    def __init__(self, img_channels=1, feature_d=64, num_classes=10):
        super(MNISTClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d),
            nn.LeakyReLU(True),

            nn.Conv2d(feature_d, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.ReLU(True),

            nn.Conv2d(feature_d * 4, 128, 3, 2, 0, bias=False),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, num_classes)  # 10개의 클래스
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)  # 0~9 각 클래스의 확률 총합이 1이 되도록한다.


# 3) 학습 준비
def one_hot_encode(labels, num_classes=10):
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)  # 만약 단일 스칼라일 경우 (배치 아님)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def show_image(img_tensor, label, pred):
    img = img_tensor.squeeze().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f"answer: {label}, prediction: {pred}")
    plt.axis('off')
    plt.show()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    sample_count = 0  # 전체 샘플 수 카운터

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        targets = one_hot_encode(labels).to(device)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 1000개마다 한 번 출력
        for i in range(len(images)):
            sample_count += 1

            if sample_count % 1000 == 0:
                predicted = torch.argmax(outputs[i]).item()
                actual = labels[i].item()
                probs = outputs[i].cpu().detach().numpy()

                print(f"[Sample {sample_count}]")
                print(f" - 실제 숫자: {actual}")
                print(f" - 예측된 숫자: {predicted}")
                print(f" - 0~9 확률: {probs.round(3)}")
                print("-" * 40)
                if predicted != actual:
                  show_image(images[i], actual, predicted)


    return total_loss / len(dataloader)

def train2(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    sample_count = 0  # 전체 샘플 수 카운터

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        targets = one_hot_encode(labels).to(device)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 틀릴 경우 출력
        for i in range(len(images)):
            sample_count += 1

            predicted = torch.argmax(outputs[i]).item()
            actual = labels[i].item()
            probs = outputs[i].cpu().detach().numpy()
            if predicted != actual:
                  print(f"[Sample {sample_count}]")
                  print(f" - 실제 숫자: {actual}")
                  print(f" - 예측된 숫자: {predicted}")
                  print(f" - 0~9 확률: {probs.round(3)}")
                  print("-" * 40)
                  show_image(images[i], actual, predicted)

    return total_loss / len(dataloader)


# 4) 하이퍼파라미터 및 실행
if __name__ == '__main__':
    root_dir = './data/train'
    batch_size = 64
    num_epochs = 100
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomMNISTDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        if epoch <= 70:
            loss = train(model, dataloader, optimizer, criterion, device)
        else:
            loss = train2(model, dataloader, optimizer, criterion, device)
        #loss = train(model, dataloader, optimizer, criterion, device)
        print("-" * 40)
        print(f"\n\n\nEpoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}\n\n\n")
        print("-" * 40)
        print("-" * 40)

# 5) 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
evaluate(model, test_loader, device)