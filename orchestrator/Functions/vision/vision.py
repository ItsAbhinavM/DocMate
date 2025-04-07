from dependencies import BaseTool
from Secrets.keys import backend_url
import uuid
import requests
import os
import json
import cv2
import face_recognition
import google.generativeai as genai
import numpy as np
from PIL import Image

# Load known faces
faces_dir = "known_faces"
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


# Function to run facial recognition model
def run_model(image):
    faces = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, faces)
    return zip(faces, encodings)


# Recognize faces and label them
def recognize_faces(frame: np.ndarray):
    detected_names = []
    for face, encoding in run_model(frame):
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        results = face_recognition.compare_faces(known_encodings, encoding, 0.5)
        for index, result in enumerate(results):
            if result:
                name = known_names[index]
                detected_names.append(name)
                # Draw a label with a name
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left, bottom + 18), font, 0.7, (255, 255, 255), 1)
    return frame, detected_names


# Capture an image from the webcam
def take_picture():
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    cam.release()
    if not ret:
        raise Exception("Failed to capture image")
    objects_description = recognize_objects(image)
    output_image, names = recognize_faces(image)
    success, encoded = cv2.imencode(".png", output_image)
    if not success:
        raise Exception("Failed to encode image")
    files = {'file': (str(uuid.uuid4()) + ".png", encoded.tobytes(), 'image/png')}
    res = requests.post(f"{backend_url}images/upload", files=files)
    url = res.json()
    return url, names, objects_description


# Convert a numpy array to a PIL image
def np_array_to_pil_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Recognize objects in the image using a generative model
def recognize_objects(image: np.ndarray):
    pil_image = np_array_to_pil_image(image)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = "What is in this image, if you detect any well known figures do not refer to them by their name merely describe their attire or figure"
    response = model.generate_content([prompt, pil_image])
    return response.text


class CamImgRecognitionTool(BaseTool):
    name = "vision"
    description = "Useful for recognizing objects, images and faces, taking pictures and using webcam/cameras"
    return_direct = True

    def _run(self, tool_input: str, **kwargs):
        print("Starting recognition")
        url, names, objects = take_picture()
        response = {"url": url, "message": "", "direct_response": True}
        if names:
            response["message"] = f"{objects}\nThe following people were found: {', '.join(names)}"
        else:
            response["message"] = f"{objects}, no familiar faces"
        return json.dumps(response)


cam_tool = CamImgRecognitionTool()
