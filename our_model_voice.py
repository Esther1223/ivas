try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import os
import sys
import subprocess
import csv
from datetime import datetime

MODEL_PATH = "obstacle_model.pth"

# ============================================================
# Logging settings
# ============================================================
ENABLE_LOG = True
LOG_PATH = "/home/username/prediction_log.csv"



# ============================================================
# Voice feedback settings
# ============================================================
ENABLE_VOICE = True

# 中文語音在 espeak 常常不完整，所以先用英文最穩
# 若你已經裝好中文語音，可把下面文字改成中文
VOICE_COMMAND = "espeak-ng"
VOICE_MAP = {
    "直走": "go straight",
    "往左": "turn left",
    "往右": "turn right",
    "往左或往右": "left or right",
    "停止": "stop",
}

last_voice_state = None
voice_process = None

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("模型載入完成")
print("類別順序：", CLASS_NAMES)

def predict_region(img):
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = prob.argmax(1).item()
        confidence = prob[0][pred].item()

    return CLASS_NAMES[pred], confidence

CENTER_SAFE_TH = 0.80
SIDE_SAFE_TH = 0.65

def apply_threshold(label, conf, region):
    if region == "center":
        th = CENTER_SAFE_TH
    else:
        th = SIDE_SAFE_TH

    if label == "safe" and conf < th:
        return "blocked"
    return label

def decide(left, center, right):
    if center == "safe":
        return "直走"
    else:
        if left == "safe" and right == "safe":
            return "往左或往右"
        elif left == "safe":
            return "往左"
        elif right == "safe":
            return "往右"
        else:
            return "停止"

def speak_decision(decision):
    """依照模型決策語音提示。只在決策改變時才講，避免一直重複。"""
    global last_voice_state, voice_process

    if not ENABLE_VOICE:
        return

    if decision == last_voice_state:
        return

    text = VOICE_MAP.get(decision, decision)
    print(f"[VOICE] {decision} -> {text}")

    try:
        # 如果上一句還沒講完，先中斷，避免語音排隊太久
        if voice_process is not None and voice_process.poll() is None:
            voice_process.terminate()

        voice_process = subprocess.Popen(
            [VOICE_COMMAND, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        last_voice_state = decision

    except FileNotFoundError:
        print(f"[VOICE] 找不到 {VOICE_COMMAND}，請先安裝：sudo apt install -y espeak-ng")
    except Exception as e:
        print(f"[VOICE] 語音播放失敗：{e}")



def save_prediction_log(mode, result, image_path="", fps=None):
    """把每次判斷結果記錄到 CSV，方便之後比對模型輸出與實際回饋。"""
    if not ENABLE_LOG:
        return

    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.exists(LOG_PATH)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "image_path": image_path,
        "left_raw": result["left"][0],
        "left_conf": f'{result["left"][1]:.4f}',
        "left_final": result["left_final"],
        "center_raw": result["center"][0],
        "center_conf": f'{result["center"][1]:.4f}',
        "center_final": result["center_final"],
        "right_raw": result["right"][0],
        "right_conf": f'{result["right"][1]:.4f}',
        "right_final": result["right_final"],
        "decision": result["decision"],
        "voice_feedback": VOICE_MAP.get(result["decision"], result["decision"]),
        "fps": "" if fps is None else f"{fps:.2f}",
    }

    fieldnames = [
        "time", "mode", "image_path",
        "left_raw", "left_conf", "left_final",
        "center_raw", "center_conf", "center_final",
        "right_raw", "right_conf", "right_final",
        "decision", "voice_feedback", "fps"
    ]

    try:
        with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[LOG] 紀錄失敗：{e}")



def split_regions_pil(img):
    w, h = img.size

    left = img.crop((0, 0, int(w * 0.45), h))
    center = img.crop((int(w * 0.25), 0, int(w * 0.75), h))
    right = img.crop((int(w * 0.55), 0, w, h))

    return left, center, right

def predict_one_image(img):
    left_img, center_img, right_img = split_regions_pil(img)

    left_result, left_conf = predict_region(left_img)
    center_result, center_conf = predict_region(center_img)
    right_result, right_conf = predict_region(right_img)

    left_final = apply_threshold(left_result, left_conf, "left")
    center_final = apply_threshold(center_result, center_conf, "center")
    right_final = apply_threshold(right_result, right_conf, "right")

    decision = decide(left_final, center_final, right_final)


    return {
        "left": (left_result, left_conf),
        "center": (center_result, center_conf),
        "right": (right_result, right_conf),

        # threshold 後真正拿來決策的結果
        "left_final": left_final,
        "center_final": center_final,
        "right_final": right_final,

        "decision": decision
    }

def run_camera():
    if Picamera2 is None:
        print("Picamera2 庫未安裝，無法使用相機模式")
        return

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(1)
    print("Camera 開啟成功，開始推論")
    if ENABLE_LOG:
        print("紀錄檔位置：", LOG_PATH)

    try:
        while True:
            start = time.time()

            frame = picam2.capture_array()
            img = Image.fromarray(frame)

            result = predict_one_image(img)
            speak_decision(result["decision"])
            fps = 1 / (time.time() - start)

            save_prediction_log("camera", result, fps=fps)

            print(
                f"L:{result['left'][0]}({result['left'][1]:.2f})->{result['left_final']} | "
                f"C:{result['center'][0]}({result['center'][1]:.2f})->{result['center_final']} | "
                f"R:{result['right'][0]}({result['right'][1]:.2f})->{result['right_final']} | "
                f"建議：{result['decision']} | FPS:{fps:.1f}"
            )

    except KeyboardInterrupt:
        print("程式停止")

    finally:
        picam2.stop()

def run_folder(folder_path):
    if not os.path.exists(folder_path):
        print("找不到資料夾：", folder_path)
        return

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_exts)
    ]
    image_files.sort()

    if len(image_files) == 0:
        print("資料夾裡沒有圖片")
        return

    print("開始測試資料夾：", folder_path)
    print("圖片數量：", len(image_files))
    if ENABLE_LOG:
        print("紀錄檔位置：", LOG_PATH)

    for idx, filename in enumerate(image_files, start=1):
        path = os.path.join(folder_path, filename)

        try:
            img = Image.open(path).convert("RGB")
            result = predict_one_image(img)
            speak_decision(result["decision"])
            save_prediction_log("folder", result, image_path=path)

            print(
                f"[{idx}/{len(image_files)}] {filename} | "
                f"L:{result['left'][0]}({result['left'][1]:.2f})->{result['left_final']} | "
                f"C:{result['center'][0]}({result['center'][1]:.2f})->{result['center_final']} | "
                f"R:{result['right'][0]}({result['right'][1]:.2f})->{result['right_final']} | "
                f"建議：{result['decision']}"
            )

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {filename} 讀取失敗：{e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請選擇模式：")
        print("  camera 模式：python3 our_model.py camera")
        print("  folder 模式：python our_model.py folder /home/username/test_images")
        sys.exit()

    mode = sys.argv[1]

    if mode == "camera":
        run_camera()

    elif mode == "folder":
        if len(sys.argv) < 3:
            print("請輸入圖片資料夾路徑")
            print("例如：python3 our_model.py folder /home/username/test_images")
        else:
            run_folder(sys.argv[2])

    else:
        print("未知模式，請用 camera 或 folder")