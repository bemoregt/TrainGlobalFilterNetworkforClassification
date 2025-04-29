import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=8, w=8):
        super().__init__()
        # rfft2는 마지막 차원을 절반으로 줄이므로 조정
        self.complex_weight = nn.Parameter(torch.randn(h, w//2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        self.h, self.w = h, w
        
    def forward(self, x):
        B, N, C = x.shape
        H, W = self.h, self.w
        
        # 2D 재구성 및 FFT 적용
        x = x.view(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # 주파수 도메인에서 가중치 적용
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        
        # 공간 도메인으로 변환
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        
        # 형태 복원
        x = x.reshape(B, N, C)
        
        return x

class GFNetBlock(nn.Module):
    def __init__(self, dim, h=8, w=8, mlp_ratio=4.):
        super().__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Global Filter
        self.filter = GlobalFilter(dim, h, w)
        
        # MLP Block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        # Global filtering with residual connection
        x = x + self.filter(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class GFNet(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, 
                 embed_dim=384, depth=12, mlp_ratio=4.):
        super().__init__()
        
        # 이미지 크기와 패치 크기 계산
        self.num_patches = (img_size // patch_size) ** 2
        h, w = img_size // patch_size, img_size // patch_size
        
        # 패치 임베딩
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            GFNetBlock(embed_dim, h, w, mlp_ratio) for _ in range(depth)
        ])
        
        # 레이어 정규화
        self.norm = nn.LayerNorm(embed_dim)
        
        # 분류 헤드
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward_features(self, x):
        B = x.shape[0]
        
        # 패치 임베딩
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # 위치 임베딩 추가
        x = x + self.pos_embed
        
        # 트랜스포머 블록 적용
        for block in self.blocks:
            x = block(x)
        
        # 정규화
        x = self.norm(x)
        
        # 글로벌 풀링
        x = x.mean(dim=1)  # 글로벌 평균 풀링
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def get_cifar10_dataloaders(batch_size=128):
    # 데이터 증강 및 정규화
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # 데이터셋 로드
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=transform_test)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'Loss': train_loss/(batch_idx+1), 'Acc': 100.*correct/total})
    
    return train_loss/len(train_loader), 100.*correct/total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'Loss': test_loss/(batch_idx+1), 'Acc': 100.*correct/total})
    
    return test_loss/len(test_loader), 100.*correct/total

def main():
    # 하이퍼파라미터 설정
    batch_size = 64  # 메모리 부하 감소를 위해 배치 크기 줄임
    epochs = 50      # 에폭 수 줄임
    lr = 1e-3
    weight_decay = 1e-4
    
    # MPS 디바이스 확인 (Apple Silicon용)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS 디바이스 사용 가능 - Apple Silicon 가속 활성화")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 디바이스 사용 가능 - GPU 가속 활성화")
    else:
        device = torch.device("cpu")
        print("CPU 디바이스 사용")
    
    # 데이터로더 가져오기
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    
    # 모델 생성 (크기 축소하여 메모리 요구사항 줄임)
    model = GFNet(
        img_size=32,           # CIFAR-10 이미지 크기
        patch_size=4,          # 패치 크기
        in_channels=3,         # RGB 채널
        num_classes=10,        # CIFAR-10 클래스 수
        embed_dim=256,         # 임베딩 차원 줄임 (384 -> 256)
        depth=8,               # 블록 수 줄임 (12 -> 8)
        mlp_ratio=4            # MLP 확장 비율
    ).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 결과 기록용 리스트
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    # 학습 루프
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 훈련
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 평가
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 모델 저장 (더 자주 저장하도록 변경)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'gfnet_cifar10_epoch{epoch+1}.pth')
    
    # 결과 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gfnet_cifar10_training.png')
    plt.show()
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'gfnet_cifar10_final.pth')
    print("Training completed!")

if __name__ == '__main__':
    main()