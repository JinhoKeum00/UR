import os

import torchvision


# MNIST 데이터셋 다운로드 및 디렉터리에 이미지 저장

def download_and_save_mnist(root_dir='./data'):

    # MNIST 훈련 데이터셋 다운로드

    train_dataset = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)

    # MNIST 테스트 데이터셋 다운로드

    test_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True)

    

    # 훈련 이미지 저장 디렉터리

    train_dir = os.path.join(root_dir, 'train')

    # 테스트 이미지 저장 디렉터리

    test_dir = os.path.join(root_dir, 'test')

    

    # 디렉터리 없으면 생성

    os.makedirs(train_dir, exist_ok=True)

    os.makedirs(test_dir, exist_ok=True)

    

    # 이미지 저장 함수

    def save_images(dataset, save_dir):

        for label in range(10):

            label_dir = os.path.join(save_dir, str(label))

            os.makedirs(label_dir, exist_ok=True)

        for idx, (image, label) in enumerate(dataset):

            # 이미지를 PNG 형태로 저장

            image_path = os.path.join(save_dir, str(label), f"img_{idx}.png")

            image.save(image_path)

    

    save_images(train_dataset, train_dir)

    save_images(test_dataset, test_dir)



# 함수 실행: MNIST 데이터를 다운로드하고 이미지로 저장합니다.

download_and_save_mnist()