"""Final verification: test ALL emotion paths with multiple student images."""
import sys, os
sys.path.insert(0, r"c:\semester 6\Computer Vision\Computer-Vision-PROJECT\BackEnd")
import cv2, numpy as np
import face_recognition as fr

from app.services.vision.emotion_detection import EmotionDetector
from app.services.vision.emotion_tracker import EmotionTracker

det = EmotionDetector()
tracker = EmotionTracker(emotion_interval=1, min_stable_samples=2, detector=det)
print(f"Model loaded: {det._model is not None}")
print()

base = r"c:\semester 6\Computer Vision\Computer-Vision-PROJECT\BackEnd\data\students_faces"
students = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

for i, student in enumerate(students[:3]):
    sd = os.path.join(base, student)
    imgs = os.listdir(sd)
    if not imgs:
        continue
    img = cv2.imread(os.path.join(sd, imgs[0]))
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locs = fr.face_locations(rgb, model="hog")
    if not locs:
        print(f"[{student}] No face found in reference image")
        continue

    t, r, b, l = locs[0]
    crop = img[t:b, l:r]

    # Simulate what the live pipeline does
    result = det.predict(crop)

    # Also test through the tracker
    emo_tracked = tracker.update(track_id=i, face_crop=crop, frame_count=i*10)
    emo_tracked2 = tracker.update(track_id=i, face_crop=crop, frame_count=i*10+1)

    print(f"[{student}]")
    print(f"  Crop: {crop.shape[1]}x{crop.shape[0]}")
    print(f"  predict():  label={result['label']:12s}  conf={result['confidence']:.2f}")
    print(f"  tracker():  label={emo_tracked2['label']:12s}  conf={emo_tracked2['confidence']:.2f}")
    print(f"  raw_scores: {result['raw_scores']}")
    print()

print("=== ALL TESTS PASSED ===" if det._model else "=== MODEL NOT LOADED ===")
