import face_recognition
import os
import pickle

def register_faces(faces_dir='faces', encoding_file='models/encodings.pkl'):
    # 切換到當前腳本所在的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 確保 models 資料夾存在
    models_dir = os.path.join(script_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"[INFO] 已建立資料夾: {models_dir}")

    known_encodings = []
    known_names = []

    # 檢查 faces 資料夾是否存在
    if not os.path.exists(faces_dir):
        print(f"[ERROR] 找不到資料夾：{faces_dir}")
        return

    for filename in os.listdir(faces_dir):
        if filename.endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[INFO] 已註冊: {name}")
            else:
                print(f"[WARNING] 無法在 {filename} 中偵測人臉")

    # 儲存編碼到 models/encodings.pkl
    with open(encoding_file, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
        print(f"[INFO] 編碼已儲存到 {encoding_file}")

if __name__ == '__main__':
    register_faces()
