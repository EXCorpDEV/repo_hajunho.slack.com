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
epochs = 50  # 컬러 이미지는 더 복잡
lr = 0.0002
latent_dim = 100
num_classes = 10
img_size = 32
channels = 3

# CIFAR-10 클래스 이름
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"🖥️  사용 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

# ============================================================================
# 데이터 로드
# ============================================================================
print("📦 CIFAR-10 데이터셋 다운로드 중...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print(f"✅ 데이터셋 로드 완료: {len(train_dataset)}개 컬러 이미지")
print(f"📋 클래스: {', '.join(class_names)}\n")

# ============================================================================
# Generator 모델 (CNN 기반 - DCGAN 스타일)
# ============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Initial projection
        self.init_size = img_size // 4  # 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size)
        )
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),  # 8 -> 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # 16 -> 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat([noise, label_input], dim=1)
        
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        
        return img

# ============================================================================
# Discriminator 모델 (CNN 기반)
# ============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 32 -> 16
            *discriminator_block(16, 32),  # 16 -> 8
            *discriminator_block(32, 64),  # 8 -> 4
            *discriminator_block(64, 128),  # 4 -> 2
        )
        
        # Output size after conv blocks
        ds_size = img_size // 2 ** 4  # 2
        
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size + num_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        label_input = self.label_emb(labels)
        d_input = torch.cat([out, label_input], dim=1)
        validity = self.adv_layer(d_input)
        
        return validity

# ============================================================================
# 모델 초기화
# ============================================================================
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 가중치 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print("🎯 모델 초기화 완료")
print(f"   Generator 파라미터: {sum(p.numel() for p in generator.parameters()):,}")
print(f"   Discriminator 파라미터: {sum(p.numel() for p in discriminator.parameters()):,}\n")

# ============================================================================
# 학습
# ============================================================================
print("🚀 학습 시작! (컬러 이미지라 시간이 좀 걸려요~)\n")

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
        
        # 실제/가짜 레이블 (Label smoothing)
        real_label = torch.ones(batch_size_current, 1, device=device) * 0.9
        fake_label = torch.zeros(batch_size_current, 1, device=device) + 0.1
        
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
        
        g_loss = criterion(discriminator(fake_imgs, gen_labels), torch.ones(batch_size_current, 1, device=device))
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
    
    # 중간 결과 시각화 (10 에폭마다)
    if (epoch + 1) % 10 == 0:
        generator.eval()
        with torch.no_grad():
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(f'Epoch {epoch+1} - CIFAR-10 생성 결과', fontsize=14, fontweight='bold')
            
            for cls in range(10):
                noise = torch.randn(1, latent_dim, device=device)
                label = torch.tensor([cls], device=device)
                generated_img = generator(noise, label)
                img = generated_img.cpu().squeeze().permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                
                row = cls // 5
                col = cls % 5
                axes[row, col].imshow(img)
                axes[row, col].set_title(class_names[cls], fontsize=10)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
        generator.train()

print("\n✅ 학습 완료!\n")

# ============================================================================
# 최종 결과 시각화
# ============================================================================
print("🎨 최종 컬러 이미지 생성 중...\n")

generator.eval()

# 각 클래스별 생성
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('CIFAR-10 GAN - 생성된 컬러 이미지들', fontsize=16, fontweight='bold')

with torch.no_grad():
    for cls in range(10):
        noise = torch.randn(1, latent_dim, device=device)
        label = torch.tensor([cls], device=device)
        
        generated_img = generator(noise, label)
        img = generated_img.cpu().squeeze().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        row = cls // 5
        col = cls % 5
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{class_names[cls]}', fontsize=11, fontweight='bold')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('cifar10_gan_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 결과 이미지 저장: cifar10_gan_results.png")

# 손실 그래프
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(G_losses, label='Generator Loss', linewidth=2, alpha=0.8)
ax.plot(D_losses, label='Discriminator Loss', linewidth=2, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('CIFAR-10 GAN Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cifar10_gan_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 손실 그래프 저장: cifar10_gan_loss.png")

# ============================================================================
# 클래스별 다양한 샘플 생성 (5개씩)
# ============================================================================
print("\n🎨 클래스별 다양한 샘플 생성 중...\n")

fig, axes = plt.subplots(10, 5, figsize=(12, 24))
fig.suptitle('CIFAR-10 GAN - 클래스별 5개 샘플', fontsize=16, fontweight='bold')

with torch.no_grad():
    for cls in range(10):
        for sample in range(5):
            noise = torch.randn(1, latent_dim, device=device)
            label = torch.tensor([cls], device=device)
            
            generated_img = generator(noise, label)
            img = generated_img.cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            
            axes[cls, sample].imshow(img)
            if sample == 0:
                axes[cls, sample].set_ylabel(class_names[cls], fontsize=11, fontweight='bold')
            axes[cls, sample].axis('off')

plt.tight_layout()
plt.savefig('cifar10_gan_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 샘플 이미지 저장: cifar10_gan_samples.png")

# ============================================================================
# 실제 vs 생성 비교
# ============================================================================
print("\n🔍 실제 이미지 vs 생성 이미지 비교...\n")

# 실제 이미지 가져오기
real_samples = []
real_labels = []
for imgs, labels in train_loader:
    real_samples.append(imgs)
    real_labels.append(labels)
    if len(real_samples) >= 1:
        break

real_samples = real_samples[0][:10].to(device)
real_labels = real_labels[0][:10].to(device)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
fig.suptitle('실제 이미지 (위) vs 생성 이미지 (아래)', fontsize=14, fontweight='bold')

with torch.no_grad():
    for i in range(10):
        # 실제 이미지
        real_img = real_samples[i].cpu().permute(1, 2, 0).numpy()
        real_img = (real_img + 1) / 2
        real_img = np.clip(real_img, 0, 1)
        axes[0, i].imshow(real_img)
        axes[0, i].set_title(f'{class_names[real_labels[i]]}', fontsize=8)
        axes[0, i].axis('off')
        
        # 생성 이미지
        noise = torch.randn(1, latent_dim, device=device)
        label = real_labels[i].unsqueeze(0)
        generated_img = generator(noise, label)
        gen_img = generated_img.cpu().squeeze().permute(1, 2, 0).numpy()
        gen_img = (gen_img + 1) / 2
        gen_img = np.clip(gen_img, 0, 1)
        axes[1, i].imshow(gen_img)
        axes[1, i].set_title(f'{class_names[real_labels[i]]}', fontsize=8)
        axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('cifar10_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 비교 이미지 저장: cifar10_comparison.png")

# ============================================================================
# 대량 샘플 그리드
# ============================================================================
print("\n🎨 100개 랜덤 샘플 생성 중...\n")

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
fig.suptitle('CIFAR-10 GAN - 100개 랜덤 샘플', fontsize=18, fontweight='bold')

with torch.no_grad():
    for i in range(100):
        noise = torch.randn(1, latent_dim, device=device)
        random_label = torch.randint(0, 10, (1,), device=device)
        
        generated_img = generator(noise, random_label)
        img = generated_img.cpu().squeeze().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        row = i // 10
        col = i % 10
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('cifar10_gan_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 그리드 이미지 저장: cifar10_gan_grid.png")
print("\n🎉 모든 작업 완료!")
