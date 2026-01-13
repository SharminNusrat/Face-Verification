import os
import random
from deepface import DeepFace

# ---------------- CONFIG ----------------
ROOT_DIR = "data/input/trainset2_0001"
MODEL_NAME = "ArcFace"
VALID_EXT = (".jpg", ".jpeg", ".png")

# Build model once
MODEL = DeepFace.build_model(MODEL_NAME)

# ---------------- VERIFY FUNCTION ----------------
def verify(img1_path, img2_path):
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=MODEL_NAME,
            model=MODEL,
            enforce_detection=False
        )
        return result["verified"]
    except Exception:
        return False

# ---------------- MAIN EVALUATION ----------------
def evaluate():
    TP = FP = TN = FN = 0

    groups = [g for g in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, g))]

    for g in groups:
        g_path = os.path.join(ROOT_DIR, g)
        subfolders = [s for s in os.listdir(g_path) if os.path.isdir(os.path.join(g_path, s))]

        # -------- SAME PERSON TEST --------
        for sub in subfolders:
            sub_path = os.path.join(g_path, sub)
            images = [f for f in os.listdir(sub_path) if f.lower().endswith(VALID_EXT)]

            if len(images) < 2:
                continue

            pivot = random.choice(images)
            pivot_path = os.path.join(sub_path, pivot)

            for img in images:
                img_path = os.path.join(sub_path, img)

                if img_path == pivot_path:
                    continue

                match = verify(pivot_path, img_path)

                if match:
                    TP += 1
                else:
                    FN += 1

        # -------- DIFFERENT PERSON TEST --------
        if len(subfolders) < 2:
            continue

        for _ in range(len(subfolders)):
            s1, s2 = random.sample(subfolders, 2)

            imgs1 = os.listdir(os.path.join(g_path, s1))
            imgs2 = os.listdir(os.path.join(g_path, s2))

            imgs1 = [f for f in imgs1 if f.lower().endswith(VALID_EXT)]
            imgs2 = [f for f in imgs2 if f.lower().endswith(VALID_EXT)]

            if not imgs1 or not imgs2:
                continue

            img1 = os.path.join(g_path, s1, random.choice(imgs1))
            img2 = os.path.join(g_path, s2, random.choice(imgs2))

            match = verify(img1, img2)

            if match:
                FP += 1
            else:
                TN += 1

    # ---------------- METRICS ----------------
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    tar = TP / (TP + FN) if (TP + FN) else 0
    far = FP / (FP + TN) if (FP + TN) else 0

    print("\n===== EVALUATION RESULT =====")
    print(f"Total Comparisons: {total}")
    print(f"TP (Correct Match): {TP}")
    print(f"FN (Missed Match): {FN}")
    print(f"TN (Correct Reject): {TN}")
    print(f"FP (False Accept): {FP}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"TAR (Recall): {tar * 100:.2f}%")
    print(f"FAR: {far * 100:.2f}%")

# ---------------- RUN ----------------
if __name__ == "__main__":
    evaluate()
