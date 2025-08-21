import sys
import io
import base64
import numpy as np
import torch
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from PIL import Image, ImageOps
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image


# ---------------------------------------------------
# Proje kökü (src'den de çalışsa köke çıksın)
# ---------------------------------------------------
Image.MAX_IMAGE_PIXELS = 20_000_000  # max 20MP resim
_THIS = Path(__file__).resolve()
project_root = _THIS.parent.parent if _THIS.parent.name == "src" else _THIS.parent
print("Project root:", project_root)

# SmallCNN için import yolu
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
from model import SmallCNN

def build_resnet18_1ch(num_classes: int):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, num_classes)
    return m

# ---------------------------------------------------
# Flask (templates: src/templates)
# ---------------------------------------------------
app = Flask(__name__, template_folder=str(_THIS.parent / "templates"))

# ---------------------------------------------------
# Cihaz
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------------------------------
# Checkpoint yükle
# ---------------------------------------------------
ckpt_path = project_root / "artifacts" / "best.pt"
print("CKPT PATH:", ckpt_path)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Model dosyası bulunamadı: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
classes = ckpt["classes"]
model_type = ckpt.get("model_type", "smallcnn")
img_size = int(ckpt.get("img_size", 28))
bin_thresh = float(ckpt.get("bin_thresh", 0.6))
print(f"Loaded ckpt meta: model_type={model_type}, img_size={img_size}, bin_thresh={bin_thresh}")

# Modeli kur
if model_type == "resnet18":
    model = build_resnet18_1ch(num_classes=len(classes)).to(device)
else:
    model = SmallCNN(num_classes=len(classes)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print("Classes (first 10):", classes[:10])

# ---------------------------------------------------
# Preprocess: crop + invert + resize(img_size) + threshold(bin_thresh) + dilation + normalize
# ---------------------------------------------------
def preprocess_image(image_bytes):
    # 1) Yükle & gri
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 2) Kenar kırp (siyah piksel < 200)
    arr0 = np.array(img)
    ys, xs = np.where(arr0 < 200)
    if len(xs) > 0 and len(ys) > 0:
        pad = 8
        x0, x1 = max(0, xs.min() - pad), min(arr0.shape[1] - 1, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(arr0.shape[0] - 1, ys.max() + pad)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # 3) Her zaman invert (canvas: beyaz zemin + siyah çizgi varsayımı)
    img = ImageOps.invert(img)

    # 4) Modelin beklediği boyuta resize
    img = img.resize((img_size, img_size), Image.BILINEAR)

    # 5) Eşik + hafif kalınlaştırma
    a = np.array(img, dtype=np.uint8)
    th = int(round(255 * bin_thresh))  # örn 0.6->153
    a = (a > th).astype(np.float32)

    # 3x3 dilation benzeri (komşulukta 1 varsa 1)
    ap = np.pad(a, 1, mode="constant")
    window_sum = (
        ap[0:-2, 0:-2] + ap[0:-2, 1:-1] + ap[0:-2, 2:] +
        ap[1:-1, 0:-2] + ap[1:-1, 1:-1] + ap[1:-1, 2:] +
        ap[2:,   0:-2] + ap[2:,   1:-1] + ap[2:,   2:]
    )
    a = (window_sum > 0).astype(np.float32)

    # 6) Normalize [-1,1]
    a = (a - 0.5) / 0.5

    # 7) Tensör [1,1,H,W]
    tensor = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return tensor.to(device)

# ---------------------------------------------------
# Tahmin (top-k)
# ---------------------------------------------------
def predict_topk(image_bytes, k=3):
    x = preprocess_image(image_bytes)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    top = prob.argsort()[-k:][::-1]
    top_list = [{"label": classes[i], "prob": float(prob[i])} for i in top]
    pred_label = classes[int(prob.argmax())]
    return pred_label, top_list

# ---------------------------------------------------
# Rotalar
# ---------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    if "image" not in body or not isinstance(body["image"], str):
        return jsonify(error="bad request"), 400
    
    try:
        header, b64data = body["image"].split(",", 1)
        raw = base64.b64decode(b64data, validate=True)
    except Exception:
        return jsonify(error="invalid image"), 400

    # Boyut sınırı (örn. 1.5MB)
    if len(raw) > 1_500_000:
        return jsonify(error="file too large"), 413

    img = Image.open(io.BytesIO(raw)).convert("L")
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])
    pred, top3 = predict_topk(image_data, k=3)
    return jsonify({"prediction": pred, "top3": top3})

# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

