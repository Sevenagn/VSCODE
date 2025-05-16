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

    # 遍歷每個子資料夾（每個人一個資料夾）
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue  # 跳過非資料夾項目

        for file in os.listdir(person_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(person_dir, file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"[INFO] 已註冊: {person_name} - {file}")
                else:
                    print(f"[WARNING] 無法在 {file} 中偵測人臉")

    # 儲存編碼
    with open(encoding_file, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
        print(f"[INFO] 編碼已儲存到 {encoding_file}")

if __name__ == '__main__':
    register_faces(r'D:\Seven\AGN\Data\face_recognition\faces')
