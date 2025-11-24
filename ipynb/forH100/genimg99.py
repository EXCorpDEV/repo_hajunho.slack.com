import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================================
# 설정
# ============================================================================
device = torch.device('cuda:0')
batch_size = 128
epochs = 20
lr = 0.0002
latent_dim = 100
num_classes = 10
img_size = 28

print(f"🖥️  사용 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

# ============================================================================
# 데이터 로드
# ============================================================================
print("📦 MNIST 데이터셋 다운로드 중...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(f"✅ 데이터셋 로드 완료: {len(train_dataset)}개 이미지\n")

# ============================================================================
# Generator 모델
# ============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat([noise, label_input], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, img_size, img_size)
        return img

# ============================================================================
# Discriminator 모델
# ============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_emb(labels)
        d_input = torch.cat([img_flat, label_input], dim=1)
        validity = self.model(d_input)
        return validity

# ============================================================================
# 모델 초기화
# ============================================================================
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print("🎯 모델 초기화 완료")
print(f"   Generator 파라미터: {sum(p.numel() for p in generator.parameters()):,}")
print(f"   Discriminator 파라미터: {sum(p.numel() for p in discriminator.parameters()):,}\n")

# ============================================================================
# 학습
# ============================================================================
print("🚀 학습 시작!\n")

G_losses = []
D_losses = []

for epoch in range(epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for i, (imgs, labels) in enumerate(pbar):
        batch_size_current = imgs.size(0)
        
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        
        # 실제/가짜 레이블
        real_label = torch.ones(batch_size_current, 1, device=device)
        fake_label = torch.zeros(batch_size_current, 1, device=device)
        
        # ============================
        # Discriminator 학습
        # ============================
        optimizer_D.zero_grad()
        
        # 실제 이미지
        real_loss = criterion(discriminator(real_imgs, labels), real_label)
        
        # 가짜 이미지
        noise = torch.randn(batch_size_current, latent_dim, device=device)
        fake_imgs = generator(noise, labels)
        fake_loss = criterion(discriminator(fake_imgs.detach(), labels), fake_label)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # ============================
        # Generator 학습
        # ============================
        optimizer_G.zero_grad()
        
        noise = torch.randn(batch_size_current, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size_current,), device=device)
        fake_imgs = generator(noise, gen_labels)
        
        g_loss = criterion(discriminator(fake_imgs, gen_labels), real_label)
        g_loss.backward()
        optimizer_G.step()
        
        # 손실 기록
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
        
        pbar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
    
    # 에폭 평균 손실
    G_losses.append(g_loss_epoch / len(train_loader))
    D_losses.append(d_loss_epoch / len(train_loader))
    
    print(f"Epoch [{epoch+1}/{epochs}] - D_loss: {D_losses[-1]:.4f}, G_loss: {G_losses[-1]:.4f}")

print("\n✅ 학습 완료!\n")

# ============================================================================
# 결과 시각화
# ============================================================================
print("🎨 이미지 생성 중...\n")

generator.eval()

# 0-9 숫자 각각 생성
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('MNIST Conditional GAN - 생성된 숫자들 (0-9)', fontsize=16, fontweight='bold')

with torch.no_grad():
    for digit in range(10):
        noise = torch.randn(1, latent_dim, device=device)
        label = torch.tensor([digit], device=device)
        
        generated_img = generator(noise, label)
        img = generated_img.cpu().squeeze().numpy()
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        
        row = digit // 5
        col = digit % 5
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'숫자 {digit}', fontsize=12)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('mnist_gan_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 결과 이미지 저장: mnist_gan_results.png")

# 손실 그래프
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(G_losses, label='Generator Loss', linewidth=2)
ax.plot(D_losses, label='Discriminator Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_gan_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 손실 그래프 저장: mnist_gan_loss.png")

# ============================================================================
# 대량 생성 샘플
# ============================================================================
print("\n🎨 다양한 샘플 생성 중...\n")

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('MNIST GAN - 64개 샘플', fontsize=16, fontweight='bold')

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(1, latent_dim, device=device)
        random_label = torch.randint(0, 10, (1,), device=device)
        
        generated_img = generator(noise, random_label)
        img = generated_img.cpu().squeeze().numpy()
        img = (img + 1) / 2
        
        row = i // 8
        col = i % 8
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'{random_label.item()}', fontsize=9)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('mnist_gan_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 샘플 이미지 저장: mnist_gan_samples.png")
print("\n🎉 모든 작업 완료!")
