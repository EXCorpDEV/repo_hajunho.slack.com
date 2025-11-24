import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================================
# 설정
# ============================================================================
device = torch.device('cuda:0')
batch_size = 128
epochs = 50
lr = 0.0002
latent_dim = 128
img_size = 64
channels = 3

print(f"🖥️  사용 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

# ============================================================================
# 합성 얼굴 데이터셋
# ============================================================================
print("🎨 합성 얼굴 패턴 데이터셋 생성 중...\n")

class SyntheticFaceDataset(Dataset):
    def __init__(self, num_samples=10000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = torch.zeros(3, self.img_size, self.img_size)
        
        # 배경 (피부색)
        skin_tone = torch.rand(1) * 0.3 + 0.5
        img += skin_tone
        
        # 얼굴 윤곽
        y, x = torch.meshgrid(torch.linspace(-1, 1, self.img_size), 
                              torch.linspace(-1, 1, self.img_size), indexing='ij')
        face_mask = ((x / 0.7) ** 2 + (y / 0.9) ** 2) < 1
        img[:, face_mask] = img[:, face_mask] * 1.1
        
        # 눈
        eye_y = torch.rand(1) * 0.2 - 0.3
        for eye_x in [-0.25, 0.25]:
            eye_mask = ((x - eye_x) / 0.08) ** 2 + ((y - eye_y) / 0.12) ** 2 < 1
            img[:, eye_mask] = 0.1
        
        # 코
        nose_y = torch.rand(1) * 0.15
        nose_mask = (torch.abs(x) < 0.05) & (y > nose_y) & (y < nose_y + 0.2)
        img[:, nose_mask] = img[:, nose_mask] * 0.9
        
        # 입
        mouth_y = torch.rand(1) * 0.1 + 0.3
        mouth_mask = (torch.abs(x) < 0.2) & (torch.abs(y - mouth_y) < 0.03)
        img[:, mouth_mask] = 0.3
        
        # 노이즈
        img += torch.randn_like(img) * 0.05
        
        img = torch.clamp(img, 0, 1)
        img = (img - 0.5) / 0.5
        
        return img, 0

train_dataset = SyntheticFaceDataset(num_samples=10000, img_size=img_size)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

print(f"✅ 합성 데이터셋 생성 완료: {len(train_dataset):,}개!\n")

# ============================================================================
# Generator
# ============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# ============================================================================
# Discriminator
# ============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

# ============================================================================
# 모델 초기화
# ============================================================================
generator = Generator().to(device)
discriminator = Discriminator().to(device)

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
print(f"   Generator: {sum(p.numel() for p in generator.parameters()):,}")
print(f"   Discriminator: {sum(p.numel() for p in discriminator.parameters()):,}\n")

fixed_noise = torch.randn(64, latent_dim, device=device)

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
    
    for i, (imgs, _) in enumerate(pbar):
        batch_size_current = imgs.size(0)
        real_imgs = imgs.to(device)
        
        real_label = torch.ones(batch_size_current, 1, device=device) * 0.9
        fake_label = torch.zeros(batch_size_current, 1, device=device) + 0.1
        
        # Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), real_label)
        noise = torch.randn(batch_size_current, latent_dim, device=device)
        fake_imgs = generator(noise)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size_current, latent_dim, device=device)
        fake_imgs = generator(noise)
        g_loss = criterion(discriminator(fake_imgs), torch.ones(batch_size_current, 1, device=device))
        g_loss.backward()
        optimizer_G.step()
        
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
        
        pbar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
    
    G_losses.append(g_loss_epoch / len(train_loader))
    D_losses.append(d_loss_epoch / len(train_loader))
    
    print(f"Epoch [{epoch+1}/{epochs}] - D: {D_losses[-1]:.4f}, G: {G_losses[-1]:.4f}")
    
    if (epoch + 1) % 10 == 0:
        generator.eval()
        with torch.no_grad():
            fake_imgs = generator(fixed_noise)
            
            fig, axes = plt.subplots(8, 8, figsize=(16, 16))
            fig.suptitle(f'Epoch {epoch+1}', fontsize=16, fontweight='bold')
            
            for idx in range(64):
                img = fake_imgs[idx].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                
                row = idx // 8
                col = idx % 8
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
        generator.train()

print("\n✅ 학습 완료!\n")

# ============================================================================
# 최종 결과
# ============================================================================
print("🎨 최종 결과 생성...\n")

generator.eval()

fig, axes = plt.subplots(8, 8, figsize=(16, 16))
fig.suptitle('생성된 얼굴 패턴 64개', fontsize=18, fontweight='bold')

with torch.no_grad():
    noise = torch.randn(64, latent_dim, device=device)
    fake_imgs = generator(noise)
    
    for idx in range(64):
        img = fake_imgs[idx].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        row = idx // 8
        col = idx % 8
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('face_gan_final.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 저장: face_gan_final.png")

# 손실 그래프
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(G_losses, label='Generator', linewidth=2)
ax.plot(D_losses, label='Discriminator', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('face_gan_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print("💾 저장: face_gan_loss.png")

# 잠재 공간 보간
print("\n🌈 잠재 공간 보간...\n")

with torch.no_grad():
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    
    steps = 10
    fig, axes = plt.subplots(1, steps, figsize=(20, 2))
    fig.suptitle('잠재 공간 보간', fontsize=14, fontweight='bold')
    
    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        img = generator(z_interp)
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'{i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('face_gan_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()

print("💾 저장: face_gan_interpolation.png")

torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'G_losses': G_losses,
    'D_losses': D_losses,
}, 'face_gan_model.pth')

print("\n🎉 모든 작업 완료!")
print(f"   최종 G Loss: {G_losses[-1]:.4f}")
print(f"   최종 D Loss: {D_losses[-1]:.4f}")
