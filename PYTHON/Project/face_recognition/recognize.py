import cv2
import face_recognition
import pickle
import time
import os
import winsound  # 僅適用於 Windows
import requests
import numpy as np

def get_avg_sharpness(cap, num_frames=5):
    sharpness_values = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values.append(sharpness)
        cv2.waitKey(30)
    return sum(sharpness_values) / len(sharpness_values) if sharpness_values else 0

def find_best_camera(max_cams=2):
    best_index = -1
    best_score = 0

    for cam_index in range(max_cams):
        cap = cv2.VideoCapture(cam_index)
        time.sleep(1.0)

        if not cap.isOpened():
            print(f"[WARN] Camera {cam_index} 無法打開")
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Camera {cam_index} 無法讀取幀")
            cap.release()
            continue

        sharpness = get_avg_sharpness(cap)
        print(f"[INFO] Camera {cam_index} sharpness: {sharpness:.2f}")

        if sharpness > best_score:
            best_score = sharpness
            best_index = cam_index

        cap.release()

    return best_index

def teamplus_postMessage(prg_id, message, subject, account, api_key ,team_sn ):
    # 團隊互動
    # https://teamplus.gce.com.tw/API/TeamService.ashx?ask=postMessage&account=va_ec6c60a8dcdc44b3b6&api_key=795c67de-e930-465d-99ef-bdeaed914ff8&team_sn=309&content_type=1&text_content=test&media_content=&file_show_name=&subject=msgtest
    api_url_TeamService = "https://teamplus.gce.com.tw/API/TeamService.ashx"
    
    hostname = os.popen('hostname').read().strip()
    message = message + "\n程式名稱:" + prg_id + "\n執行主機:" + hostname

    params_TeamService = {
        "ask": "postMessage",
        "account": account,
        "api_key": api_key,
        "team_sn": team_sn,
        "content_type": "1",
        "text_content": message,
        "media_content": "",
        "file_show_name": "",
        "subject": subject
    }
    try:
        # # 发送POST请求到API，并传递参数
        response = requests.post(api_url_TeamService, params_TeamService)

        # 检查响应状态码
        if response.status_code == 200:
            # 如果响应成功，解析JSON数据
            data = response.json()
            # 处理数据...
            # print(data)
            if data["IsSuccess"] == True :
                print("\n\nTeam+團隊互動发送成功\n\n")
            else:
                print("\n\nTeam+團隊互動发送失败\n\n")
        else:
            print("Error:", response.status_code)

    except requests.RequestException as e:
        print("Error:", e)


# 自動定位當前腳本所在的目錄，避免相對路徑錯誤
script_dir = os.path.dirname(os.path.abspath(__file__))
encoding_path = os.path.join(script_dir, 'models', 'encodings.pkl')

# 載入已知人臉
with open(encoding_path, 'rb') as f:
    known_encodings, known_names = pickle.load(f)

# 特定要監控的人名（可以是多個）
# target_persons = ["003296", "005546", "013529", "022652", "028572", "065200", "071349", "075134", "087985", "088652"]  # <== 修改這裡
target_persons = ["087985"]  # <== 修改這裡
# target_persons = ["043266"]  # <== 修改這裡

notified = False  # 是否已通知
cooldown_time = 30  # 通知冷卻時間（秒）
# last_notify_time = 0
last_notify_time = {}  # key = name, value = timestamp
# 控制是否提醒未知人臉
alert_unknown = False  # True 表示提醒未註冊的人，False 表示不提醒

#0 代表第一個攝像頭（默認）,1 代表第二個攝像頭
best_cam = find_best_camera()
video = cv2.VideoCapture(best_cam)
# 設置解析度
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 寬度
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # 高度

# 設置幀率（FPS）
# video.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] 開始辨識，按 q 離開")

# 初始化 FPS 計算用變數
prev_time = time.time()
while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 預設 model='hog' 是 CPU-only
    # face_locations = face_recognition.face_locations(rgb_frame)
    
    # cnn gpu，識別率高，幀率低
    # face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 縮圖找臉（省時間），人臉識別率略低，但幀率快
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
    small_locations = face_recognition.face_locations(small_frame, model='cnn')
    # 回推原圖座標
    face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in small_locations]
    # 在原圖做 face encoding（保留細節）
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # matches = face_recognition.compare_faces(known_encodings, face_encoding)
        # 使用較嚴格的比對容差
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        if True in matches:
            matched_indexes = [i for i, m in enumerate(matches) if m]
            for matched_idx in matched_indexes:
                name = known_names[matched_idx]

                # === 偵測到目標人物（多個支援）===
                if name in target_persons:
                    now = time.time()
                    # if not notified or (now - last_notify_time > cooldown_time):
                    # 檢查該人物是否需要冷卻
                    if name not in last_notify_time or (now - last_notify_time[name] > cooldown_time):
                        print(f"[ALERT] 偵測到 {name}！")
                        winsound.Beep(1500, 500)

                        # 發送 Team+ 通知
                        prg_id = "face_recognition"
                        subject = "📵危險"
                        message = f"偵測到目標人物：{name}。"
                        account = "va_596bdbc869e442e889"
                        api_key = "2532f614-9de4-4dfe-9fa4-ce5ee119d3aa"
                        team_sn = "366"
                        teamplus_postMessage(prg_id, message, subject, account, api_key, team_sn)

                        notified = True
                        # last_notify_time = now
                        last_notify_time[name] = now
        else:
            # 未註冊人臉
            if alert_unknown:
                now = time.time()
                # if not notified or (now - last_notify_time > cooldown_time):
                if "Unknown" not in last_notify_time or (now - last_notify_time["Unknown"] > cooldown_time):
                    print(f"[ALERT] 偵測到未知人員！")
                    winsound.Beep(1000, 500)  

                    # 調用發送 Team+ 訊息的函數
                    prg_id = "face_recognition"  # 你的程式名稱
                    subject = "⚠️警告"
                    message = "偵測到未知人員！"
                    account = "va_596bdbc869e442e889"
                    api_key = "2532f614-9de4-4dfe-9fa4-ce5ee119d3aa"
                    team_sn = "366"
                    teamplus_postMessage(prg_id, message, subject, account, api_key, team_sn)

                    notified = True
                    # last_notify_time = now
                    last_notify_time["Unknown"] = now

        # 顯示辨識結果
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    # ==========================
    # 計算 FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 顯示 FPS 到畫面上
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # ==========================
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
