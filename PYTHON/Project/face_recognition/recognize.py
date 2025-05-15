import cv2
import face_recognition
import pickle
import time
import os
import winsound  # 僅適用於 Windows

# 自動定位當前腳本所在的目錄，避免相對路徑錯誤
script_dir = os.path.dirname(os.path.abspath(__file__))
encoding_path = os.path.join(script_dir, 'models', 'encodings.pkl')

# 載入已知人臉
with open(encoding_path, 'rb') as f:
    known_encodings, known_names = pickle.load(f)

# 特定要監控的人名（可以是多個）
target_persons = ["Seven", "Jack", "Alice"]  # <== 修改這裡

notified = False  # 是否已通知
cooldown_time = 3  # 通知冷卻時間（秒）
last_notify_time = 0
# 控制是否提醒未知人臉
alert_unknown = True  # True 表示提醒未註冊的人，False 表示不提醒

video = cv2.VideoCapture(0)
print("[INFO] 開始辨識，按 q 離開")

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

            # === 偵測到目標人物（多個支援）===
            if name in target_persons:
                now = time.time()
                if not notified or (now - last_notify_time > cooldown_time):
                    print(f"[ALERT] 偵測到 {name}！")
                    winsound.Beep(1000, 500)  # 1000 Hz, 0.5 秒
                    notified = True
                    last_notify_time = now
        else:
            # 未註冊人臉
            if alert_unknown:
                now = time.time()
                if not notified or (now - last_notify_time > cooldown_time):
                    print(f"[ALERT] 偵測到未知人員！")
                    winsound.Beep(1500, 500)  # 更高頻率表示未知
                    notified = True
                    last_notify_time = now

        # 顯示辨識結果
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
