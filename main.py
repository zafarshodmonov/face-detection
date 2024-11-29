# Import
import cv2
import dlib
import numpy as np
import time
import math

def get_landmarks(image, face_detector, landmark_predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        landmarks = landmark_predictor(gray, faces[0])
        return [(p.x, p.y) for p in landmarks.parts()]
    return None

def get_eye_centers(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    left_center = (sum([p[0] for p in left_eye]) // 6, sum([p[1] for p in left_eye]) // 6)
    right_center = (sum([p[0] for p in right_eye]) // 6, sum([p[1] for p in right_eye]) // 6)
    return left_center, right_center

def get_rotation_angle(left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_image(image, angle, center):
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated



# video_capture = cv2.VideoCapture(0)

# start_time o'zgaruvchisi e'lon qilinadi va hozirgi vaqt olingan
start_time = time.time()

# step_time vaqtini belgilash (sekundlarda, masalan, 5 sekund)
step_time = 120

# # # step_time o'tib o'tmaganini tekshirish
# # while True:
# #     ret, frame = video_capture.read()
# #     if not ret:
# #         break
# #     current_time = time.time()
# #     elapsed_time = current_time - start_time

# #     if elapsed_time >= step_time:
# #         break
# #     else:
# #         pass

# #     landmarks = get_landmarks(frame, detector, predictor)

# #     if landmarks:
# #         left_eye, right_eye = get_eye_centers(landmarks)
# #         angle = get_rotation_angle(left_eye, right_eye)
        
# #         # Yuzni aylantirish markazi ikki ko‘z orasidagi o‘rtacha nuqta
# #         center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
# #         frame = rotate_image(frame, -angle, center)
    
# #     cv2.imshow("Aligned Face", frame)

# #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     # faces = detector(gray)
    

# #     # # Yuzlarni belgilash
# #     # for face in faces:
# #     #     x, y, w, h = face.left(), face.top(), face.width(), face.height()
# #     #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
# #     #     cropped_face = frame[y:y+h, x:x+w]
# #     #     # Landmarklarni olish
# #     #     landmarks = predictor(gray, face)
# #     #     landmarks_array = np.array([[landmarks.part(i).x - x, landmarks.part(i).y - y] for i in range(68)])

# #     #     resized_face = cv2.resize(cropped_face, (224, 224))
# #     #     # Landmarklarni mos o'lchamga transformatsiya qilish
# #     #     scale_x = 224 / w
# #     #     scale_y = 224 / h
# #     #     resized_landmarks = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in landmarks_array])
# #     #     for (x, y) in resized_landmarks:
# #     #         cv2.circle(resized_face, (x, y), 2, (0, 255, 0), -1)

# #     #cv2.imshow(f"Resized Face {idx+1}", face)
# #     # cv2.imshow("Video", frame)
# #     # cv2.imshow("Croped Face", cropped_face)
# #     # cv2.imshow("Resized Face", resized_face)
# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break

# # video_capture.release()
# # cv2.destroyAllWindows()

# import cv2
# import dlib
# import numpy as np

# # Yuz aniqlash uchun dlib modeli
# face_detector = dlib.get_frontal_face_detector()

# # Parametrlar: Yuz va quti o‘lchamlari
# target_width, target_height = 128, 128  # Yuz o‘lchami
# output_width, output_height = 256, 256  # Quti o‘lchami

# def get_face_box(image, face_detector):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_detector(gray)
#     if len(faces) > 0:
#         x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
#         return (x, y, w, h)
#     return None

# def resize_face(image, face_box, target_width, target_height):
#     x, y, w, h = face_box
#     face = image[y:y+h, x:x+w]  # Yuzni kesib olish
#     resized_face = cv2.resize(face, (target_width, target_height))  # O‘lchamni o‘zgartirish
#     return resized_face

# def place_face_in_box(resized_face, output_width, output_height):
#     output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)  # Bo‘sh quti
#     start_x = (output_width - resized_face.shape[1]) // 2
#     start_y = (output_height - resized_face.shape[0]) // 2
#     output_image[start_y:start_y+resized_face.shape[0], start_x:start_x+resized_face.shape[1]] = resized_face
#     return output_image

# # Video ochish
# cap = cv2.VideoCapture(0)  # Web-kamera yoki video yo‘li

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     face_box = get_face_box(frame, face_detector)
#     if face_box:
#         resized_face = resize_face(frame, face_box, target_width, target_height)
#         output_image = place_face_in_box(resized_face, output_width, output_height)
#     else:
#         # Agar yuz topilmasa, bo‘sh tasvir ko‘rsatamiz
#         output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
#     # Natijani ko‘rsatish
#     cv2.imshow("Aligned Face in Box", output_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# Yuz aniqlash va landmarklar uchun model
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Parametrlar
target_width, target_height = 128, 128
output_width, output_height = 256, 256

def calculate_rotation_angle(landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def align_face(image, face_box, landmarks):
    x, y, w, h = face_box
    face_region = image[y:y+h, x:x+w]
    angle = calculate_rotation_angle(landmarks)
    image_center = (face_region.shape[1] // 2, face_region.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    aligned_face = cv2.warpAffine(face_region, rotation_matrix, (face_region.shape[1], face_region.shape[0]))
    return aligned_face

def resize_face(face_image, target_width, target_height):
    resized_face = cv2.resize(face_image, (target_width, target_height))
    return resized_face

def place_face_in_box(resized_face, output_width, output_height):
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    start_x = (output_width - resized_face.shape[1]) // 2
    start_y = (output_height - resized_face.shape[0]) // 2
    output_image[start_y:start_y+resized_face.shape[0], start_x:start_x+resized_face.shape[1]] = resized_face
    return output_image

# Video orqali jarayon
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    elapsed_time = current_time - start_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if elapsed_time >= step_time:
        break
    else:
        pass

    if len(faces) > 0:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #cv2.line(frame, (0, x + w // 2), (y, x +w // 2), (255, 0, 0), 1)
        landmarks = landmark_predictor(gray, face)
        face_box = (face.left(), face.top(), face.width(), face.height())
        
        aligned_face = align_face(frame, face_box, landmarks)
        resized_face = resize_face(aligned_face, target_width, target_height)
        output_image = place_face_in_box(resized_face, output_width, output_height)
    else:
        output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    cv2.imshow("Aligned Face in Box", output_image)
    cv2.imshow("Orginal Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
