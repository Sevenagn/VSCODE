import face_recognition
import os
import pickle

def register_known_faces(known_faces_dir=r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\PYTHON\Project\face_recognition\known_faces', encoding_file='encodings.pkl'):
    known_encodings = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[INFO] 已註冊: {name}")
            else:
                print(f"[WARNING] 無法在 {filename} 中偵測人臉")

    with open(encoding_file, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
        print(f"[INFO] 編碼已儲存到 {encoding_file}")

if __name__ == '__main__':
    register_known_faces()
