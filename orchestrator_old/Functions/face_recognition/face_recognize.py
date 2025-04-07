import os
import sys
import cv2
import face_recognition
import time
import numpy


def benchmark(func):
    def wrapper(*arg, **kw):
        """source: http://www.daniweb.com/code/snippet368.html"""
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f"Function took {t2-t1} Seconds to run")
        return res

    return wrapper


# Load known faces
faces_dir = os.path.join(sys.path[0], "known_faces")
known_encodings = []
known_names = []
for img in os.listdir(faces_dir):
    img_path = os.path.join(faces_dir, img)
    img_name = img.split(".")[0]

    img_np = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img_np)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(img_name)
    else:
        print(f"Error! No faces detected in {img_name}!")


# @benchmark
def run_model(image):
    # Make image 1/2th for faster face recognition
    image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    faces = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, faces)
    return zip(faces, encodings)


# Takes an input image and returns an image with faces outlined and labelled
# as well as a list of the people detected
def recognize_faces(frame: numpy.array):
    detected_names = []
    for face, encoding in run_model(frame):
        face = [val * 2 for val in face]  # Restore image dimensions
        top, right, bot, left = face
        cv2.rectangle(frame, (left, top), (right, bot), (0, 0, 255), 2)
        results = face_recognition.compare_faces(known_encodings, encoding)
        for index, result in enumerate(results):
            if result:
                name = known_names[index]
                detected_names.append(name)
                # Draw a label with a name
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    frame, name, (left, bot + 18), font, 0.7, (255, 255, 255), 1
                )
    return frame, detected_names


if __name__ == "__main__":
    import requests

    res = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/f/f4/ObamNSASpeech2.PNG"
    )
    frame = cv2.imdecode(numpy.frombuffer(res.content, numpy.uint8), cv2.IMREAD_COLOR)
    frame, _ = recognize_faces(frame)
    cv2.imshow("r", frame)
    cv2.waitKey(0)

    # video_capture = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = video_capture.read()
    #     frame, _ = recognize_faces(frame)
    #     cv2.imshow("face", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break
