import argparse, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from model import SmallCNN

def load_model(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallCNN(num_classes=len(ckpt['classes']))
    model.load_state_dict(ckpt['model']); model.to(device); model.eval()
    return model, ckpt['classes'], device

def preprocess(img_path):
    img = Image.open(img_path).convert('L').resize((28,28))
    arr = (np.array(img).astype('float32')/255.0)[None,None,...]
    return torch.from_numpy(arr)

def main(args):
    model, classes, device = load_model(args.checkpoint)
    x = preprocess(args.image).to(device)
    t0=time.perf_counter(); 
    with torch.no_grad(): out = model(x); dt=(time.perf_counter()-t0)*1000
    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    idx = int(prob.argmax()); print(f"pred={classes[idx]} prob={prob[idx]:.3f} time_ms={dt:.1f}")
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', default='artifacts/best.pt')
    main(ap.parse_args())
