import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox
import datetime

# 保證 faces 存在於腳本目錄下
script_dir = os.path.dirname(os.path.abspath(__file__))
# SAVE_DIR = os.path.join(script_dir, "faces")
SAVE_DIR = r"D:\Seven\AGN\Data\face_recognition\faces"
MAX_IMAGES = 20  # 每人最多抓取圖像數量
CAPTURE_INTERVAL = 1  # 每秒拍一張（秒）

def capture_faces():
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("輸入錯誤", "請輸入姓名！")
        return

    person_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    last_capture_time = 0
    messagebox.showinfo("提示", "按下 Q 鍵停止拍攝")

    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # 控制每秒只儲存一次
        if faces is not None and len(faces) > 0 and (current_time - last_capture_time >= CAPTURE_INTERVAL):
            for (x, y, w, h) in faces:
                count += 1
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                # filename = os.path.join(person_dir, f"{count}.jpg")
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S_%f")  # %f 是微秒，6位數
                filename = os.path.join(person_dir, f"{timestamp}.jpg")

                cv2.imwrite(filename, face_img)
                last_capture_time = current_time
                break  # 只存一張臉（如要多臉拍多張可移除此行）

        # 顯示畫面與統計
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Images: {count}/{MAX_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("完成", f"{name} 的臉部圖像已儲存於 {person_dir}")

root = tk.Tk()
# root.withdraw()  # 隐藏主窗口
root.geometry("260x120")
root.title("人臉收集")
root.resizable(width=False,height=False)

label_text = tk.Label(root,text="姓名:")
label_text.grid(row=0, column=0, padx=10, pady=30)

name_entry = tk.Entry(root, width=20)
name_entry.grid(row=0, column=1, padx=10, pady=30)

start_btn = tk.Button(root,text="開始",command=capture_faces)
start_btn.grid(row=2, column=0, columnspan=2, pady=0)

root.mainloop()
