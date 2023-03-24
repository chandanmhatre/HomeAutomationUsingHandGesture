import jsonlines
import os
import json
import tensorflow as tf
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from keypoint_classification import clearTfModel, run_keypoint_classification
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import streamlit as st
from datetime import datetime, timedelta
from streamlit_modal import Modal
import streamlit.components.v1 as components
import h5py


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.8,
    )
    args = parser.parse_args()
    return args


def main(gestureNumber, loggingMode, viewOnly, automationOn):
    with st.spinner(""):
        # Argument parsing #################################################################
        args = get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        use_brect = True

        # Camera preparation ###############################################################
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0]
                                          for row in keypoint_classifier_labels]
        keypoint_classifier_labels = getGestureLabel()
        with open(
            "model/point_history_classifier/point_history_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)

        #  ########################################################################
        mode = 0
        # present_time = datetime.now()
        # present_time = present_time.strftime('%H:%M:%S')
        global maxCount
        updated_time = datetime.now() + timedelta(seconds=15)
        updated_time = updated_time.strftime("%H:%M:%S")
        maxCount = updated_time.split(":")[2]
        with st.empty():
            while checkCondition(viewOnly, updated_time):
                fps = cvFpsCalc.get()
                number, mode = gestureNumber, loggingMode
                # Camera capture #####################################################
                ret, image = cap.read()
                if not ret:
                    break
                # image = cv.bilateralFilter(image, 5, 50, 100)
                image = cv.flip(image, 1)  # Mirror display
                debug_image = copy.deepcopy(image)

                # Detection implementation #############################################################
                image.flags.writeable = False
                image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)

                # ###########
                # ret, thresh = cv.threshold(cv.cvtColor(
                #     image, cv.COLOR_BGR2GRAY), 127, 255, 0)
                # M = cv.moments(thresh)
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                # # print(cX)
                # # print(cY)
                # ####################
                # if cX > 200 and cX < 400:
                #     print('middel')
                # if cX > 1 and cX < 200:
                #     print('left')
                # if cX > 400 and cX < 600:
                #     print('rigth')

                results = hands.process(image)
                image.flags.writeable = True

                #  ####################################################################
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(
                            debug_image, hand_landmarks)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list
                        )
                        pre_processed_point_history_list = pre_process_point_history(
                            debug_image, point_history
                        )
                        # Write to the dataset file
                        logging_csv(
                            number,
                            mode,
                            pre_processed_landmark_list,
                            pre_processed_point_history_list,
                        )

                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(
                            pre_processed_landmark_list)
                        if hand_sign_id == "Not required":  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])
                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(
                            pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list
                            )
                        print(f'hand_sign_id:{hand_sign_id}')
                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history
                        ).most_common()
                        # Drawing part
                        debug_image = draw_bounding_rect(
                            use_brect, debug_image, brect)
                        debug_image = draw_landmarks(
                            debug_image, landmark_list)
                        if (hand_sign_id != 100):
                            if automationOn:
                                print(keypoint_classifier_labels[hand_sign_id])
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                handedness,
                                keypoint_classifier_labels[hand_sign_id],
                                point_history_classifier_labels[most_common_fg_id[0][0]],
                            )
                        else:
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                handedness,
                                '',
                                point_history_classifier_labels[most_common_fg_id[0][0]],
                            )
                else:
                    point_history.append([0, 0])

                debug_image = draw_point_history(debug_image, point_history)
                debug_image = draw_info(
                    debug_image, fps, mode, number, viewOnly)
                st.image(debug_image, use_column_width=True, channels="BGR")
            global isLoggingComplete
            isLoggingComplete = True
            cap.release()
            cv.destroyAllWindows()


def checkCondition(viewOnly, updated_time):
    if (viewOnly):
        return True
    else:
        return datetime.now().strftime("%H:%M:%S") != updated_time


# def select_mode(key, mode):
#     number = -1
#     if 0 <= key <= 9:  # 0 ~ 9
#         number = key
#     if key == 110:  # n
#         mode = 0
#     if key == 107:  # k
#         mode = 1
#     if key == 104:  # h
#         mode = 2
#     return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(
            landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(
            landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(
            landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(
            landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(
            landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(
                landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(
                landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(
                landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(
                landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(
                landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(
                landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(
                landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(
                landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(
                landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(
            landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(
            landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(
            landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(
            landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(
                landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(
                landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(
                landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]),
                      8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]),
                      8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]),
                      8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]),
                      8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]),
                      5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]),
                      8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]),
                     (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]),
                 (brect[2], brect[1] - 22), (0, 0, 0), -1)
    print(brect)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 +
                int(index / 2), (152, 251, 152), 2
            )

    return image


def draw_info(image, fps, mode, number, viewOnly):
    rmeainingTime = str(int(maxCount) - int(datetime.now().strftime("%S")))
    if (viewOnly == False):
        cv.putText(
            image,
            "Remaining Time : " + rmeainingTime,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
    # cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
    #            1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ["Logging Key Point", "Logging Point History"]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image


def getGestureLabel():
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0]
                                      for row in keypoint_classifier_labels]
    return keypoint_classifier_labels


state = st.session_state
st.title("Home automation using hand gesture")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create Sidebar
st.sidebar.title("Sidebar")
st.sidebar.subheader("Parameter")

app_mode = st.sidebar.selectbox("App Mode", ["Set Gesture", "Video", "About"])
if "disabled" not in state:
    state["disabled"] = False


def disable():
    state["disabled"] = True


def enable():
    state["disabled"] = False


def openModal(onClickYes):
    title = '<p style = "font-family: Courier;color: black;font-weight: bolder;font-size: larger;" >Gesture detection complete </p>'
    modal = Modal(title, key="modal")
    open_modal = st.button("Open")
    if open_modal:
        modal.open()
    if modal.is_open():
        with modal.container():
            label = '<p style = "font-family: Courier;color: blue;font-weight: bolder;font-size: medium;text-align: center;" >Are you want continue to next process ?</p>'
            st.markdown(label, unsafe_allow_html=True)
            # st.image(image, channels="BGR")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pass
            with col2:
                yesBtn = st.button("Yes")
                if yesBtn:
                    onClickYes
            with col3:
                noBtn = st.button("No")
                if noBtn:
                    modal.close()
            with col4:
                pass


# Add Sidebar and Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# About Page
if app_mode == "About":
    enable()
    state["disabled"] = False
    st.markdown(
        """
                ## ABOUT\n
                In this application we are using **MediaPipe** for a hand gesture detection. **StreamLit** is used to create the Web Graphical User Interface (GUI) \n
    """
    )
    option = st.selectbox("Saved Gestures", getGestureLabel())
    st.write("You selected:", option)


jsonFile = 'gesture-pin.csv'


def writeJson(user_word, selectedPin):
    isDuplicatePin = False
    with open(jsonFile, 'r') as read:
        file = csv.reader(read, delimiter=',')
        lines = list(file)
        for obj in lines:
            if str(selectedPin) == str(obj[1]):
                isDuplicatePin = True
                st.error(f'Pin {selectedPin} is already selected for {obj[0]}')
                break
    if isDuplicatePin == False:
        with open(jsonFile, 'a', newline="", encoding="utf-8") as append:
            file = csv.writer(append)
            file.writerow([user_word, selectedPin])
    return isDuplicatePin


if app_mode == "Set Gesture":
    numbers = list(itertools.chain(range(1, 21)))
    isLoggingComplete = False
    with st.form(key="Form1", clear_on_submit=False):
        with st.sidebar:
            user_word = st.sidebar.text_input(
                "Enter Gesture Name", disabled=state.disabled, on_change=disable
            )
            selectedPin = st.sidebar.selectbox(
                "Select Pin", numbers
            )
    if user_word in getGestureLabel():
        st.error(f'{user_word} already exit')
        enable()
        changeBtn = st.button('change name')
        if changeBtn:
            enable()
    else:
        if user_word != "" and state["disabled"] and writeJson(user_word, selectedPin) == False:
            csv_path = "model/keypoint_classifier/keypoint_classifier_label.csv"
            with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([user_word])
            st.button("Cancel", on_click=enable, use_container_width=True)
            main(len(getGestureLabel()), 1, False, False)
            if isLoggingComplete:
                run_keypoint_classification()


# Video Page
if app_mode == "Video":
    enable()
    if len(getGestureLabel()) != 0:
        checked = st.checkbox('automation on')
        main(len(getGestureLabel()), 0, True, checked)
    else:
        st.title("no data found")
