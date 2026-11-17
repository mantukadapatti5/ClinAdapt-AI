import base64, io, json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import models
LABELS = ["Pneumonia", "Pneumothorax", "Effusion"]
WEIGHTS = os.path.join("..", "ml", "saved_models", "densenet121_clinadapt.pt")
CALIB = os.path.join("..", "ml", "saved_models", "calibration.json")
MEAN = np.array([0.485,0.456,0.406],dtype=np.float32); STD = np.array([0.229,0.224,0.225],dtype=np.float32)
net = models.densenet121(weights=None); net.classifier = torch.nn.Linear(1024, 3)
TRAINED = os.path.exists(WEIGHTS)
if TRAINED: net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
net.eval()
TEMP = json.load(open(CALIB)).get("temperature", 1.0) if os.path.exists(CALIB) else 1.0
VERSION = "densenet121-trained" if TRAINED else "densenet121-UNTRAINED-demo"

def _prep(img):
    a = (np.asarray(img.convert("RGB").resize((224,224)),dtype=np.float32)/255 - MEAN)/STD
    return torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0)

def predict(img: Image.Image) -> dict:
    x = _prep(img)
    feats = net.features(x); feats.retain_grad()
    logits = net.classifier(F.adaptive_avg_pool2d(F.relu(feats),1).flatten(1))
    probs = torch.sigmoid(logits/TEMP)[0].detach().numpy()
    k = int(probs.argmax()); logits[0,k].backward()
    cam = F.relu((feats.grad.mean((2,3),keepdim=True)*feats).sum(1))[0].detach().numpy()
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    cam = np.asarray(Image.fromarray((cam*255).astype("uint8")).resize(img.size, Image.BILINEAR),dtype=np.float32)/255
    base = np.asarray(img.convert("RGB"),dtype=np.float32)
    heat = np.stack([cam*255, cam*90, np.zeros_like(cam)],-1)
    over = (0.6*base + 0.4*heat*1).clip(0,255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(over).save(buf,"PNG")
    findings = [{"label": l, "score": round(float(p),4),
                 "uncertain": bool(0.35 <= p <= 0.65)} for l,p in zip(LABELS, probs)]
    return {"findings": findings, "heatmap_label": LABELS[k], "heatmap_png": base64.b64encode(buf.getvalue()).decode(),
            "model_version": VERSION, "trained": TRAINED, "temperature": TEMP}
