import numpy as np
from PIL import Image
def screen(img: Image.Image) -> dict:
    """Technical screening only - does NOT certify clinical adequacy."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = g[1:-1,1:-1]*4 - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]
    m = {"blur_score": round(float(lap.var()),1), "brightness": round(float(g.mean()),1), "contrast": round(float(g.std()),1),
         "width": img.width, "height": img.height}
    issues = []
    if min(img.size) < 256: issues.append("resolution below 256 px")
    if m["blur_score"] < 15: issues.append("image may be blurred")
    if m["brightness"] < 40 or m["brightness"] > 215: issues.append("brightness out of range")
    if m["contrast"] < 20: issues.append("low contrast")
    if not 0.6 < img.width/img.height < 1.6: issues.append("unusual aspect ratio")
    return {"status": "suitable" if not issues else "flagged", "issues": issues, "metrics": m}
