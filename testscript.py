import os
import cv2
import random
import numpy as np
import logging
from collections import defaultdict

# ===============================
# 1. ENVIRONMENT & LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("InsightFaceEval")

# Import InsightFace
try:
    from insightface.app import FaceAnalysis
except ImportError:
    logger.error("InsightFace not installed! Run: pip install insightface onnxruntime")
    exit(1)

# ===============================
# 2. CONFIG
# ===============================
ROOT_DIR = "testset"
OUTPUT_BASE = "./evaluation_results"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Production-grade threshold (strict for banking scenarios)
SIMILARITY_THRESHOLD = 0.45  # Cosine similarity (higher = stricter)
# Typical banking systems use 0.4-0.5 for high security

# Create output folders
OUTPUT_FOLDERS = {
    "TP": os.path.join(OUTPUT_BASE, "TP_true_positive"),
    "FN": os.path.join(OUTPUT_BASE, "FN_false_negative"),
    "TN": os.path.join(OUTPUT_BASE, "TN_true_negative"),
    "FP": os.path.join(OUTPUT_BASE, "FP_false_positive")
}

for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# ===============================
# 3. INITIALIZE INSIGHTFACE
# ===============================
logger.info("Loading InsightFace buffalo_l model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("Model loaded successfully!")

# ===============================
# 4. HELPER FUNCTIONS
# ===============================
def resize_to_height(img, target_height):
    """Resize image maintaining aspect ratio"""
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (new_width, target_height))

def get_embedding(img_path):
    """Extract face embedding using InsightFace"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load: {img_path}")
            return None
        
        faces = app.get(img)
        if len(faces) == 0:
            logger.warning(f"No face detected: {img_path}")
            return None
        
        # Use the first (most confident) face
        return faces[0].embedding
    
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def save_comparison(img1_path, img2_path, category, score, person1, person2):
    """Save side-by-side comparison image with detailed naming"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return
    
    # Resize to same height
    img1_resized = resize_to_height(img1, 512)
    img2_resized = resize_to_height(img2, 512)
    
    # Create side-by-side image
    combined = np.hstack((img1_resized, img2_resized))
    
    # Create descriptive filename
    # Format: {person1}_{person2}_score{score}_randomID.jpg
    random_id = random.randint(100000, 999999)
    filename = f"{person1}_vs_{person2}_sim{score:.4f}_{random_id}.jpg"
    
    output_path = os.path.join(OUTPUT_FOLDERS[category], filename)
    cv2.imwrite(output_path, combined)

# ===============================
# 5. EVALUATION LOGIC
# ===============================
def process_subfolder(group, subfolder, subfolder_path, all_subfolders):
    """Process one subfolder for genuine and imposter tests"""
    
    # Get all images in this subfolder
    try:
        images = [f for f in os.listdir(subfolder_path) 
                  if f.lower().endswith(VALID_EXTENSIONS)]
    except:
        return 0, 0, 0, 0
    
    if len(images) < 2:
        return 0, 0, 0, 0
    
    TP = FN = TN = FP = 0
    
    # --- CASE 1: GENUINE TEST (Same Person) ---
    # Pick one pivot image
    pivot_name = random.choice(images)
    pivot_path = os.path.join(subfolder_path, pivot_name)
    pivot_emb = get_embedding(pivot_path)
    
    if pivot_emb is None:
        return 0, 0, 0, 0
    
    # Compare with all other images in same subfolder
    for img_name in images:
        if img_name == pivot_name:
            continue
        
        img_path = os.path.join(subfolder_path, img_name)
        img_emb = get_embedding(img_path)
        
        if img_emb is None:
            continue
        
        similarity = cosine_similarity(pivot_emb, img_emb)
        is_match = similarity >= SIMILARITY_THRESHOLD
        
        if is_match:
            TP += 1
            # Optionally save TP (uncomment if you want all TPs saved)
            # save_comparison(pivot_path, img_path, "TP", similarity, subfolder, subfolder)
        else:
            FN += 1
            # Save FALSE NEGATIVE (same person but didn't match)
            save_comparison(pivot_path, img_path, "FN", similarity, subfolder, subfolder)
    
    # --- CASE 2: imposter TEST (Different Person) ---
    # Find a different subfolder from the same group
    other_subfolders = [s for s in all_subfolders if s != subfolder]
    
    if len(other_subfolders) == 0:
        return TP, FN, TN, FP
    
    # Pick one random different subfolder
    imposter_subfolder = random.choice(other_subfolders)
    imposter_path = os.path.join(os.path.dirname(subfolder_path), imposter_subfolder)
    
    try:
        imposter_images = [f for f in os.listdir(imposter_path) 
                          if f.lower().endswith(VALID_EXTENSIONS)]
    except:
        imposter_images = []
    
    # Limit imposter tests to 5 per subfolder
    imposter_images = random.sample(imposter_images, min(5, len(imposter_images)))
    
    for imp_img_name in imposter_images:
        imp_img_path = os.path.join(imposter_path, imp_img_name)
        imp_emb = get_embedding(imp_img_path)
        
        if imp_emb is None:
            continue
        
        similarity = cosine_similarity(pivot_emb, imp_emb)
        is_match = similarity >= SIMILARITY_THRESHOLD
        
        if is_match:
            FP += 1
            # Save FALSE POSITIVE (different people but matched)
            save_comparison(pivot_path, imp_img_path, "FP", similarity, 
                          subfolder, imposter_subfolder)
        else:
            TN += 1
            # Optionally save TN (uncomment if you want all TNs saved)
            # save_comparison(pivot_path, imp_img_path, "TN", similarity, 
            #               subfolder, imposter_subfolder)
    
    return TP, FN, TN, FP

# ===============================
# 6. MAIN EVALUATION
# ===============================
def main():
    if not os.path.exists(ROOT_DIR):
        logger.error(f"Path '{ROOT_DIR}' not found!")
        return
    
    total_TP = total_FN = total_TN = total_FP = 0
    
    logger.info(f"Starting evaluation with threshold: {SIMILARITY_THRESHOLD}")
    logger.info(f"Results will be saved to: {OUTPUT_BASE}")
    
    # Iterate through groups (0001, 0002, ...)
    groups = sorted([g for g in os.listdir(ROOT_DIR) 
                     if os.path.isdir(os.path.join(ROOT_DIR, g))])
    
    for group in groups:
        group_path = os.path.join(ROOT_DIR, group)
        
        # Get all subfolders in this group
        subfolders = sorted([s for s in os.listdir(group_path) 
                           if os.path.isdir(os.path.join(group_path, s))])
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(group_path, subfolder)
            
            TP, FN, TN, FP = process_subfolder(group, subfolder, subfolder_path, subfolders)
            
            total_TP += TP
            total_FN += FN
            total_TN += TN
            total_FP += FP
            
            # Progress update
            total_ops = total_TP + total_FN + total_TN + total_FP
            acc = (total_TP + total_TN) / total_ops * 100 if total_ops > 0 else 0
            
            print(f"Processing {group}/{subfolder} | Acc: {acc:.1f}% | "
                  f"TP:{total_TP} FN:{total_FN} TN:{total_TN} FP:{total_FP}   ", 
                  end="\r")
    
    # ===============================
    # FINAL METRICS
    # ===============================
    genuine = total_TP + total_FN
    imposter = total_TN + total_FP
    total = genuine + imposter
    
    print("\n\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Threshold: {SIMILARITY_THRESHOLD} (cosine similarity)")
    print("-" * 50)
    print(f"True Positives  (TP): {total_TP:6d}  [Same person, matched]")
    print(f"False Negatives (FN): {total_FN:6d}  [Same person, NOT matched]")
    print(f"True Negatives  (TN): {total_TN:6d}  [Diff person, NOT matched]")
    print(f"False Positives (FP): {total_FP:6d}  [Diff person, matched]")
    print("=" * 50)
    
    if total > 0:
        accuracy = (total_TP + total_TN) / total * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
    
    if genuine > 0:
        tar = total_TP / genuine * 100
        print(f"TAR (True Accept Rate / Recall): {tar:.2f}%")
    
    if imposter > 0:
        far = total_FP / imposter * 100
        print(f"FAR (False Accept Rate): {far:.2f}%")
    
    print("=" * 50)
    print(f"\nMismatch images saved to: {OUTPUT_BASE}")
    print(f"  - FN (missed same person): {OUTPUT_FOLDERS['FN']}")
    print(f"  - FP (wrongly matched diff person): {OUTPUT_FOLDERS['FP']}")
    print("=" * 50)

if __name__ == "__main__":
    main()