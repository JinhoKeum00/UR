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


# 2) 클래스 (Encoder + Decoder)
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1,16,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),

        nn.Conv2d(16,32,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(64,128,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),

        nn.Conv2d(128,256,3,padding=1),
        nn.ReLU()
    )

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0),-1)
    return out

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.layer1 = nn.Sequential(
        nn.ConvTranspose2d(256,128,3,2,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),

        nn.ConvTranspose2d(128,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
    )
    self.layer2 = nn.Sequential(
        nn.ConvTranspose2d(64,16,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(16),

        nn.ConvTranspose2d(16,1,3,2,1,1),
        nn.ReLU()
    )

  def forward(self,x):
    out = x.view(x.size(0),256,7,7)
    out = self.layer1(out)
    out = self.layer2(out)
    return out


# 3) 학습 준비
def train(encoder, decoder, dataloader, optimizer, loss_func, device):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        outputs = encoder(images)
        outputs = decoder(outputs)

        loss = loss_func(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(dataloader)


# 4) 하이퍼파라미터 및 실행
if __name__ == '__main__':
    root_dir = './data/train'
    batch_size = 256
    num_epochs = 10
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomMNISTDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    parameters = list(encoder.parameters())+list(decoder.parameters())

    optimizer = optim.Adam(parameters, lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        loss = train(encoder, decoder, dataloader, optimizer, loss_func, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    torch.save(encoder.state_dict(), "encoder_weights.pth")
    print("Encoder weights saved to encoder_weights.pth")

#----------------------------------------------------------------------------
# Head 클래스 추가 및 head 훈련 함수 추가
class Head(nn.Module):
    def __init__(self, input_dim=256*7*7, num_classes=10):
        super(Head, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

def train_head(encoder, head, dataloader, optimizer, loss_func, device):
    head.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            features = encoder(images)

        outputs = head(features)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# 학습 준비 및 진행
if __name__ == '__main__':
    root_dir = './data/train'
    batch_size = 256
    num_epochs = 10
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomMNISTDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("encoder_weights.pth"))
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    head = Head().to(device)

    optimizer_head = optim.Adam(head.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        avg_loss, accuracy = train_head(encoder, head, dataloader, optimizer_head, loss_func, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(head.state_dict(), "head_weights.pth")
    print("Head weights saved to head_weights.pth")

#----------------------------------------------------------------------------
# Encoder + Head 동시 학습 준비
def train_EH(encoder, head, dataloader, optimizer, loss_func, device):
    encoder.train()
    head.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = encoder(images)
        outputs = head(outputs)

        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# 학습 준비 및 진행
if __name__ == '__main__':
    root_dir = './data/train'
    batch_size = 256
    num_epochs = 10
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomMNISTDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(device)
    head = Head().to(device)

    parameters = list(encoder.parameters())+list(head.parameters())

    optimizer_EH = optim.Adam(parameters, lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        avg_loss, accuracy = train_EH(encoder, head, dataloader, optimizer_EH, loss_func, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(encoder.state_dict(), "encoder_weights.pth")
    print("Encoder weights saved to encoder_weights.pth")

    torch.save(head.state_dict(), "head_weights.pth")
    print("Head weights saved to head_weights.pth")

#----------------------------------------------------------------------------
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