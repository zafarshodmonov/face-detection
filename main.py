import cv2
import dlib
import numpy as np

# 1. Frontal yuzlarni aniqlovchi obyektni yaratish
detector = dlib.get_frontal_face_detector()

# 2. Rasmni yuklash
img = cv2.imread('images/image_1.jpg')

# 3. Rang formatini BGR'dan RGB'ga o'zgartirish
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Yuzlarni aniqlash
faces = detector(gray)

# # 5. Aniqlangan yuzlarni konsolda chiqarish
# print(f"Aniqlangan yuzlar soni: {len(faces)}")
# for i, face in enumerate(faces):
#     print("left: ", face.left())
#     print("top: ", face.top())
#     print("right: ", face.right())
#     print("bottom: ", face.bottom())
#     print("width: ", face.width())
#     print("height: ", face.height())
#     print("area: ", face.area())
#     print("contains(500, 500): ", face.contains(500, 500))
#     print("is_empty(): ", face.is_empty())
#     print("==================================================")


# # Dlib modeli yuklash
# detector = dlib.get_frontal_face_detector()  # Yuzni aniqlash modeli
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark modeli


# Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_face = frame[y:y+h, x:x+w]
        #landmarks = predictor(gray, face)
        #landmarks_array = np.array([[landmarks.part(i).x - x, landmarks.part(i).y - y] for i in range(68)])
        #resized_face = cv2.resize(cropped_face, (224, 224))
        #scale_x = 224 / w
        #scale_y = 224 / h
        #resized_landmarks = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in landmarks_array])
        # for (x, y) in resized_landmarks:
        #     cv2.circle(resized_face, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imshow("", cropped_face)
        cv2.waitKey(0)
        #cap.release()
        cv2.destroyAllWindows()

