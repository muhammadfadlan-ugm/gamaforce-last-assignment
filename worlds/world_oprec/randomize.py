import cv2, json, random, argparse, os

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.path.join(here, "..", "models", "obstacles"))
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--out", default="target.json")
a = ap.parse_args()

seed = a.seed if a.seed is not None else random.randrange(10**6)
marker_id = random.Random(seed).choice([1, 2, 3])

tex = os.path.join(a.model_dir, "materials", "textures")
os.makedirs(tex, exist_ok=True)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#img = cv2.aruco.generateImageMarker(d, marker_id, 600)

if hasattr(cv2.aruco, "generateImageMarker"):
    img = cv2.aruco.generateImageMarker(d, marker_id, 600)
else:
    img = cv2.aruco.drawMarker(d, marker_id, 600)   # OpenCV < 4.7i

mg = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite(os.path.join(tex, "marker_slot1.png"), img)

json.dump({"seed": seed, "marker_id": marker_id, "target_bullseye": marker_id},
          open(a.out, "w"))
print(f"seed={seed} marker={marker_id} -> TARGET bullseye {marker_id} (saved to {a.out})")
