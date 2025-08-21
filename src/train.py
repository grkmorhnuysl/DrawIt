# train.py — QuickDraw (numpy_bitmap) için güçlü ve bilgi veren eğitim betiği
# - Varsayılan: ResNet18 (1-kanal), img_size=64, OneCycleLR, AMP (isteğe bağlı), label smoothing
# - Augment: binarize (bin_thresh), hafif dilation, affine (rot+shift), az gürültü
# - MixUp/CutMix opsiyonel
# - Early stopping, tqdm, ayrıntılı log
# - En iyi model: artifacts/best.pt  (app.py checkpoint meta’yı otomatik okur)

import argparse, math, time, random
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.models import resnet18
from tqdm import tqdm

# ---------------------------
# Yardımcılar
# ---------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_indices(y, train=0.8, val=0.1, test=0.1, seed=42):
    y = np.array(y)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(np.zeros_like(y), y))
    ytmp = y[tmp_idx]
    ratio = test / (val + test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    va_rel, te_rel = next(sss2.split(np.zeros_like(ytmp), ytmp))
    va_idx = tmp_idx[va_rel]; te_idx = tmp_idx[te_rel]
    return tr_idx, va_idx, te_idx

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

# ---------------------------
# Dataset
# ---------------------------
class QuickDrawNPY(Dataset):
    """
    Beklenen dizin: data_root/bitmap/{class}.npy
    Her dosya: (N,784) veya (N,28,28) uint8. __getitem__: x->[1,28,28] in [0,1], y->int
    """
    def __init__(self, data_root: str, class_names: List[str], per_class_limit: int | None = None):
        self.class_names = class_names
        self.samples = []
        self.labels = []
        root = Path(data_root) / "bitmap"
        print(f"[DATA] Klasör: {root}")
        for ci, name in enumerate(class_names):
            f = root / f"{name}.npy"
            if not f.exists():
                raise FileNotFoundError(f"[DATA] Dosya yok: {f}")
            arr = np.load(f, mmap_mode="r")  # (N,784) veya (N,28,28)
            if per_class_limit:
                arr = arr[:per_class_limit]
            if arr.ndim == 2:
                arr = arr.reshape(-1, 28, 28)
            arr = (arr.astype("float32") / 255.0)
            self.samples.append(arr)
            self.labels += [ci] * len(arr)
        self.samples = np.concatenate(self.samples, axis=0)  # (M, 28, 28)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"[DATA] Toplam örnek: {len(self.labels)}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]).unsqueeze(0), int(self.labels[idx])  # [1,28,28]

# ---------------------------
# Augment & Preprocess (torch)
# ---------------------------
class TorchAugment:
    def __init__(self, img_size=64, bin_thresh=0.6, thick_prob=0.4, affine_deg=10, translate=0.06, noise_std=0.01):
        self.img_size = img_size
        self.bin_thresh = bin_thresh
        self.thick_prob = thick_prob
        self.affine_deg = affine_deg
        self.translate = translate
        self.noise_std = noise_std

    @staticmethod
    def _dilate3x3(x):  # hafif kalınlaştırma
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _resize(self, x, size):
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    def _affine(self, x):
        B, _, H, W = x.shape
        deg = (torch.rand(B, device=x.device) * 2 - 1) * self.affine_deg
        tx  = (torch.rand(B, device=x.device) * 2 - 1) * self.translate
        ty  = (torch.rand(B, device=x.device) * 2 - 1) * self.translate
        rad = deg * math.pi / 180.0
        cos, sin = torch.cos(rad), torch.sin(rad)
        theta = torch.zeros((B, 2, 3), device=x.device, dtype=x.dtype)
        theta[:,0,0] = cos; theta[:,0,1] = -sin; theta[:,1,0] = sin; theta[:,1,1] = cos
        theta[:,0,2] = tx * 2.0; theta[:,1,2] = ty * 2.0
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def __call__(self, x, train=True):
        # x: [B,1,28,28] in [0,1]
        if self.img_size != 28:
            x = self._resize(x, self.img_size)
        if train:
            if self.affine_deg > 0 or self.translate > 0:
                x = self._affine(x)
            x = (x > self.bin_thresh).float()
            if random.random() < self.thick_prob:
                x = self._dilate3x3(x)
            if self.noise_std > 0:
                x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)
        x = (x - 0.5) / 0.5  # [-1,1]
        return x

# ---------------------------
# Modeller
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

def build_resnet18_1ch(num_classes: int):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, num_classes)
    return m

# ---------------------------
# MixUp / CutMix
# ---------------------------
def mixup_cutmix(x, y, alpha=0.3, mode="mixup"):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[idx]
        return x, (y, y[idx], lam), "mixup"
    else:
        B, C, H, W = x.shape
        rx = np.random.uniform(0, W); ry = np.random.uniform(0, H)
        rw = W * np.sqrt(1 - lam); rh = H * np.sqrt(1 - lam)
        x1 = int(np.clip(rx - rw/2, 0, W)); x2 = int(np.clip(rx + rw/2, 0, W))
        y1 = int(np.clip(ry - rh/2, 0, H)); y2 = int(np.clip(ry + rh/2, 0, H))
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return x, (y, y[idx], lam), "cutmix"

def criterion_with_aux(criterion, out, targets):
    if isinstance(targets, tuple):
        y1, y2, lam = targets
        return lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
    else:
        return criterion(out, targets)

# ---------------------------
# Eğitim / Değerlendirme
# ---------------------------
def train_one_epoch(model, loader, device, optimizer, scaler, scheduler, criterion, aug,
                    mix_alpha=0.0, mix_mode="mixup", grad_clip=1.0):
    model.train()
    losses = []; correct = 0; total = 0
    pbar = tqdm(loader, desc="  [Train]", ncols=100)
    use_cuda = (device.type == "cuda")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        x = aug(x, train=True)
        targets = y
        if mix_alpha > 0:
            x, y_mix, _ = mixup_cutmix(x, y, alpha=mix_alpha, mode=mix_mode)
            targets = y_mix

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu",
                                 enabled=(scaler is not None and use_cuda)):
            out = model(x)
            loss = criterion_with_aux(criterion, out, targets)
        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()  # OneCycleLR: batch başına

        losses.append(loss.item())
        if mix_alpha <= 0:
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{(correct/total*100):.1f}%")
        else:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
    train_acc = (correct / total) if total else float("nan")
    return float(np.mean(losses)), train_acc

@torch.no_grad()
def evaluate(model, loader, device, aug):
    model.eval()
    preds = []; ys = []
    pbar = tqdm(loader, desc="  [Val/Test]", ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        x = aug(x, train=False)
        out = model(x)
        preds += out.argmax(1).cpu().tolist()
        ys += y.cpu().tolist()
    acc = accuracy_score(ys, preds)
    return acc, (ys, preds)

# ---------------------------
# Main
# ---------------------------
def main(args):
    print("========== QuickDraw Trainer ==========")
    seed_everything(args.seed)

    # Yollar
    _this = Path(__file__).resolve()
    project_root = _this.parents[1] if _this.parent.name == "src" else _this.parent
    classes_path = (project_root / args.classes).resolve()
    data_dir = (project_root / args.data).resolve()
    artifacts_dir = (project_root / "artifacts"); artifacts_dir.mkdir(exist_ok=True)
    best_path = artifacts_dir / "best.pt"

    print(f"[PATH] Project root : {project_root}")
    print(f"[PATH] Classes file : {classes_path}")
    print(f"[PATH] Data dir     : {data_dir}")
    print(f"[PATH] Save to      : {best_path}")

    # FAST preset (isteğe bağlı)
    if args.fast:
        print("[FAST] Hızlı preset aktif (CPU-dostu).")
        args.model = "smallcnn"
        args.img_size = 28
        args.epochs = min(args.epochs, 6)
        args.bs = min(args.bs, 256)
        args.per_class_limit = args.per_class_limit or 1500
        args.mix_alpha = 0.0
        args.affine_deg = 6
        args.translate = 0.04
        args.noise_std = 0.0
        if args.num_workers == 0:
            args.num_workers = 4
        args.pin_memory = True
        args.bin_thresh = 0.55  # biraz daha yumuşak

    # Sınıflar
    if not classes_path.exists(): raise FileNotFoundError(classes_path)
    classes = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    num_classes = len(classes)
    print(f"[INFO] Classes ({num_classes}): {classes[:10]}{' ...' if num_classes>10 else ''}")
    if num_classes < 2: raise ValueError("En az iki sınıf gerekli.")

    # Dataset & split
    print("[INFO] Dataset yükleniyor...")
    ds = QuickDrawNPY(str(data_dir), classes, per_class_limit=args.per_class_limit)
    y = ds.labels
    tr_idx, va_idx, te_idx = split_indices(y, train=args.train_ratio, val=args.val_ratio, test=args.test_ratio, seed=args.seed)
    tr_set = torch.utils.data.Subset(ds, tr_idx)
    va_set = torch.utils.data.Subset(ds, va_idx)
    te_set = torch.utils.data.Subset(ds, te_idx)
    print(f"[SPLIT] Train: {len(tr_set)} | Val: {len(va_set)} | Test: {len(te_set)}")

    # Sampler (opsiyonel)
    if args.weighted_sampler:
        class_sample_count = np.bincount(y[tr_idx], minlength=num_classes)
        class_weights = 1.0 / np.clip(class_sample_count, 1, None)
        weights = class_weights[y[tr_idx]]
        sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=len(weights), replacement=True)
        shuffle = False
        print("[INFO] WeightedRandomSampler aktif.")
    else:
        sampler = None; shuffle = True

    # DataLoader hız ayarları
    pin = args.pin_memory and torch.cuda.is_available()
    nw = args.num_workers
    pf = args.prefetch_factor if nw and args.prefetch_factor else None
    pw = True if nw and nw > 0 else False

    tr_loader = DataLoader(tr_set, batch_size=args.bs, shuffle=shuffle, sampler=sampler,
                           num_workers=nw, pin_memory=pin, prefetch_factor=pf, persistent_workers=pw)
    va_loader = DataLoader(va_set, batch_size=args.bs, shuffle=False,
                           num_workers=nw, pin_memory=pin, prefetch_factor=pf, persistent_workers=pw)
    te_loader = DataLoader(te_set, batch_size=args.bs, shuffle=False,
                           num_workers=nw, pin_memory=pin, prefetch_factor=pf, persistent_workers=pw)

    # Augmenter
    img_size = args.img_size
    aug = TorchAugment(img_size=img_size, bin_thresh=args.bin_thresh, thick_prob=args.thick_prob,
                       affine_deg=args.affine_deg, translate=args.translate, noise_std=args.noise_std)
    print(f"[AUG] img_size={img_size}, bin_thresh={args.bin_thresh}, thick_prob={args.thick_prob}, "
          f"affine_deg={args.affine_deg}, translate={args.translate}, noise_std={args.noise_std}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "resnet18":
        model = build_resnet18_1ch(num_classes)
    else:
        model = SmallCNN(num_classes=num_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.model} | Parametre: {n_params:,} | Cihaz: {device}")
    if args.model == "resnet18" and img_size != 64:
        print(f"[NOTE] ResNet18 için genelde img_size=64 önerilir (şu an {img_size}).")

    # Optimizasyon
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    print(f"[OPT] lr={args.lr}, wd={args.wd}, label_smoothing={args.label_smoothing}, amp={bool(scaler)}")
    print(f"[RUN] epochs={args.epochs}, bs={args.bs}, mix_alpha={args.mix_alpha} ({args.mix_mode}), early_stop={args.early_stop}")

    # Eğitim
    best_val = 0.0
    no_improve = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n[{epoch}/{args.epochs}] Epoch başlıyor...")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, tr_loader, device, optimizer, scaler, scheduler, criterion, aug,
            mix_alpha=args.mix_alpha, mix_mode=args.mix_mode, grad_clip=args.grad_clip
        )
        val_acc, _ = evaluate(model, va_loader, device, aug)
        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        print(f"[{epoch}/{args.epochs}] loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | lr={current_lr:.2e} | epoch_time={human_time(dt)}")

        if val_acc > best_val:
            best_val = val_acc; no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "model_type": args.model,
                "img_size": img_size,
                "bin_thresh": args.bin_thresh
            }, best_path)
            print(f"  ✅ En iyi model kaydedildi: {best_path} (Val_Acc={best_val:.4f})")
        else:
            no_improve += 1
            print(f"  ℹ️  Gelişme yok ({no_improve}/{args.early_stop}).")
            if no_improve >= args.early_stop:
                print(f"  ⛔ Early stopping: {epoch}. epokta durdu.")
                break

    # Test
    print("\n[TEST] En iyi model yükleniyor ve test ediliyor...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    test_acc, (ty, tp) = evaluate(model, te_loader, device, aug)
    print(f"[TEST] ACC: {test_acc:.4f}")
    cm = confusion_matrix(ty, tp)
    print("[TEST] CONFUSION MATRIX:\n", cm)
    print("[TEST] CLASSIFICATION REPORT:\n", classification_report(ty, tp, target_names=classes, digits=4))
    for i, cls in enumerate(classes):
        acc_cls = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  - {cls}: {acc_cls:.2%}")

    total_time = time.time() - t_start
    print(f"\n[TOTAL] Eğitim süresi: {human_time(total_time)}")
    print("=======================================")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Yollar
    p.add_argument("--data", type=str, default="data/quickdraw")
    p.add_argument("--classes", type=str, default="classes.txt")

    # Eğitim
    p.add_argument("--model", type=str, choices=["smallcnn", "resnet18"], default="resnet18")
    p.add_argument("--epochs", type=int, default=20)         # makul varsayılan
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--per_class_limit", type=int, default=4000)  # kalite için makul limit

    # Veri/augment
    p.add_argument("--img_size", type=int, default=64)       # ResNet18 için öneri
    p.add_argument("--bin_thresh", type=float, default=0.60)
    p.add_argument("--thick_prob", type=float, default=0.40)
    p.add_argument("--affine_deg", type=float, default=10.0)
    p.add_argument("--translate", type=float, default=0.06)
    p.add_argument("--noise_std", type=float, default=0.01)

    # Mix augment
    p.add_argument("--mix_alpha", type=float, default=0.2)   # 0 kapatır
    p.add_argument("--mix_mode", type=str, choices=["mixup", "cutmix"], default="mixup")

    # Optimizasyon
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=6)
    p.add_argument("--weighted_sampler", action="store_true")

    # Split/seed
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # DataLoader hız
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")

    # Hızlı preset
    p.add_argument("--fast", action="store_true", help="Hızlı (CPU-dostu) preset")

    args = p.parse_args()
    main(args)
