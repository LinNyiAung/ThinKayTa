from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import copy

# Import your classifiers and utils
from app import calc_landmark_list, pre_process_landmark, pre_process_point_history
from model import KeyPointClassifier, PointHistoryClassifier
from utils import *

app = Flask(__name__)

# Initialize models and variables
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

@app.route('/detect', methods=['POST'])
def detect_hand_sign():
    file = request.files['frame'].read()
    np_img = np.fromstring(file, np.uint8)
    image = cv.imdecode(np_img, cv.IMREAD_COLOR)

    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    response = {
        "gesture": None
    }

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            pre_processed_point_history_list = pre_process_point_history(
                debug_image, point_history)
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == history_length * 2:
                finger_gesture_id = point_history_classifier(
                    pre_processed_point_history_list)

            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(
                finger_gesture_history).most_common()

            response["gesture"] = most_common_fg_id[0][0]

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
