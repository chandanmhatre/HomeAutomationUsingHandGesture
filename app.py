import csv
import itertools
import os
import pickle
import time
import cv2
import numpy as np
from raspberry_firebase_admin import FirbaseServices
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components


class mpHands:
    import mediapipe as mp

    def __init__(self, maxHands=1, model_complexity=1, tol1=.5, tol2=.5):
        self.hands = self.mp.solutions.hands.Hands(
            False, maxHands, model_complexity, tol1, tol2)

    def Marks(self, frame):
        myHands = []
        results = self.hands.process(frame)
        if results.multi_hand_landmarks != None:
            for handLandMarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                global brect
                brect = calc_bounding_rect(frame, handLandMarks)
                global landmark_list
                global handSide
                handSide = handedness.classification[0].label[0:]
                landmark_list = calc_landmark_list(
                    frame, handLandMarks)
                myHand = []
                for landMark in handLandMarks.landmark:
                    myHand.append(
                        (int(landMark.x*width), int(landMark.y*height)))
                myHands.append(myHand)
        return myHands


mainModelPath = 'MainModel/main.pkl'
gesturePinCsvPath = 'model/gesture-pin.csv'
gestureNamesCsvPath = 'model/gesture-names.csv'
detectionKeypointLength = 10
firbaseService = FirbaseServices()
findHands = mpHands()


def main(gestureName, isTraining, automationOn):
    #  isTraining = 1 to Train, 0 to Recognize
    with st.spinner(""):
        global width, height
        width = 1280
        height = 720
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        time.sleep(2)
        keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
        train = isTraining
        if train == 1:
            knownGestures = []
            gestNames = [gestureName]
            trainModelName = gestureName+'.pkl'
        if train == 0:
            with open(mainModelPath, 'rb') as f:
                gestNames = pickle.load(f)
                knownGestures = pickle.load(f)
        tol = 10
        with st.empty():
            while checkCondition(isTraining, knownGestures):
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (width, height))
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                handData = findHands.Marks(frame)
                if train == 1:
                    cv2.putText(
                        frame,
                        "Remaining : " +
                        str(detectionKeypointLength - len(knownGestures)),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA
                    )
                    if handData != []:
                        time.sleep(1)
                        knownGesture = findDistances(handData[0])
                        knownGestures.append(knownGesture)

                if train == 0:
                    if handData != []:
                        unknownGesture = findDistances(handData[0])
                        detectedGesture = findGesture(
                            unknownGesture, knownGestures, keyPoints, gestNames, tol)
                        cv2.rectangle(frame, (brect[0], brect[1]),
                                      (brect[2], brect[1] - 30), (0, 0, 0), -1)
                        cv2.rectangle(frame, (brect[0], brect[1]),
                                      (brect[2], brect[3]), (0, 0, 0), 1)
                        cv2.putText(
                            frame,
                            f"{handSide} : {detectedGesture}",
                            (brect[0] + 5, brect[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                        if automationOn:
                            gesturepinDataList = getGesturePin()
                            for gesture in gesturepinDataList:
                                serverOnOff = firbaseService.getGesturePinDetails()
                                if gesture[0] == detectedGesture:
                                    selectedPin = gesture[1]
                                    onOrOff = True if serverOnOff[str(
                                        selectedPin)] == False else False
                                    firbaseService.updatePinValue(
                                        str(selectedPin), onOrOff)
                                    time.sleep(2)
                for hand in handData:
                    for ind in keyPoints:
                        frame = draw_landmarks(
                            frame, landmark_list)
                st.image(frame, use_column_width=True)
            if train == 1:
                with open('model/'+trainModelName, 'wb') as f:
                    pickle.dump(gestNames, f)
                    pickle.dump(knownGestures, f)
                newGestNames = []
                newKnownGestures = []
                for d in os.listdir("model/"):
                    if d.endswith('.pkl'):
                        with open("model/"+d, 'rb') as f:
                            gestNames2 = pickle.load(f)
                            knownGestures2 = pickle.load(f)
                            for data in gestNames2:
                                newGestNames.append(data)
                            for data in knownGestures2:
                                newKnownGestures.append(data)
                with open(mainModelPath, 'wb') as out:
                    pickle.dump(
                        newGestNames, out, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(
                        newKnownGestures, out, protocol=pickle.HIGHEST_PROTOCOL)
                global isLoggingComplete
                isLoggingComplete = True


def checkCondition(isTraining, knownGestures):
    if (isTraining == 0):
        return True
    else:
        return len(knownGestures) < detectionKeypointLength


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

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


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(
            landmark_point[3]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[3]), tuple(
            landmark_point[4]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(
            landmark_point[6]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[6]), tuple(
            landmark_point[7]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[7]), tuple(
            landmark_point[8]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv2.line(
            image, tuple(landmark_point[9]), tuple(
                landmark_point[10]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[10]), tuple(
                landmark_point[11]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[11]), tuple(
                landmark_point[12]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv2.line(
            image, tuple(landmark_point[13]), tuple(
                landmark_point[14]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[14]), tuple(
                landmark_point[15]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[15]), tuple(
                landmark_point[16]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv2.line(
            image, tuple(landmark_point[17]), tuple(
                landmark_point[18]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[18]), tuple(
                landmark_point[19]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[19]), tuple(
                landmark_point[20]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(
            landmark_point[1]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[1]), tuple(
            landmark_point[2]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[2]), tuple(
            landmark_point[5]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv2.line(image, tuple(landmark_point[5]), tuple(
            landmark_point[9]), (0, 0, 0), 6)
        cv2.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[9]), tuple(
                landmark_point[13]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[13]), tuple(
                landmark_point[17]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv2.line(
            image, tuple(landmark_point[17]), tuple(
                landmark_point[0]), (0, 0, 0), 6
        )
        cv2.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv2.circle(image, (landmark[0], landmark[1]),
                       8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv2.circle(image, (landmark[0], landmark[1]),
                       8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]),
                       8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]),
                       8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]),
                       5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]),
                       8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def findDistances(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    palmSize = ((handData[0][0]-handData[9][0])**2 +
                (handData[0][1]-handData[9][1])**2)**(1./2.)
    for row in range(0, len(handData)):
        for column in range(0, len(handData)):
            distMatrix[row][column] = (((handData[row][0]-handData[column][0])**2+(
                handData[row][1]-handData[column][1])**2)**(1./2.))/palmSize
    return distMatrix


def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error = error+abs(gestureMatrix[row]
                              [column]-unknownMatrix[row][column])
    return error


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol):
    errorArray = []
    for i in range(0, len(gestNames) * detectionKeypointLength, 1):
        error = findError(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
    minIndex = 0
    errorSplitList = list(split(errorArray, len(gestNames)))
    for i in range(0, len(errorSplitList), 1):
        if errorMin in errorSplitList[i]:
            minIndex = i
    if errorMin < tol:
        gesture = gestNames[minIndex]
        # print('++++++++++++++++++++++++++++')
        # print(errorMin)
        # print('--------------------')
        # print(tol)
        # print('--------------------')
        # print(gesture)
    if errorMin >= tol:
        gesture = 'Unknown'
    return gesture


def getGestureNames():
    with open(gestureNamesCsvPath, encoding="utf-8-sig"
              ) as f:
        gesture_labels = csv.reader(f)
        gesture_labels = [row for row in gesture_labels]
    return gesture_labels


def getGesturePin():
    with open(gesturePinCsvPath, encoding="utf-8-sig"
              ) as f:
        gesture_pin_labels = csv.reader(f)
        gesture_pin_labels = [row for row in gesture_pin_labels]
    return gesture_pin_labels


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


def checkDuplicatePinAndUpdatePin(user_word, selectedPin):
    isDuplicatePin = False
    with open(gesturePinCsvPath, 'r') as read:
        file = csv.reader(read, delimiter=',')
        lines = list(file)
        lines = list(filter(None, lines))
        for obj in lines:
            if str(selectedPin) == str(obj[1]):
                isDuplicatePin = True
                for pin in plugPinDict:
                    if plugPinDict[pin] == selectedPin:
                        pinName = pin
                        break
                st.error(
                    f'"{pinName}" plug is already selected for gesture "{obj[0]}"')
                break
    if isDuplicatePin == False:
        with open(gesturePinCsvPath, 'a', newline="", encoding="utf-8") as append:
            file = csv.writer(append)
            file.writerow([user_word, selectedPin])
    return isDuplicatePin


if app_mode == "Set Gesture":
    global plugPinDict
    plugPinDict = firbaseService.getPlugPinDetails()
    sortedPluPinList = sorted(plugPinDict)
    isLoggingComplete = False
    with st.form(key="Form1", clear_on_submit=False):
        with st.sidebar:
            user_word = st.sidebar.text_input(
                "Enter Gesture Name", disabled=state.disabled, on_change=disable
            )
            selectedPinName = st.sidebar.selectbox(
                "Select Plug", sortedPluPinList
            )
    selectedPin = plugPinDict[selectedPinName]
    if user_word in getGestureNames():
        st.error(f'{user_word} already exit')
        enable()
        changeBtn = st.button('change name')
        if changeBtn:
            enable()
    else:
        if user_word != "" and state["disabled"] and checkDuplicatePinAndUpdatePin(user_word, selectedPin) == False:
            st.button("Cancel", on_click=enable, use_container_width=True)
            main(user_word, 1, False)
            if isLoggingComplete:
                complete = False
                with open(gestureNamesCsvPath, "a", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([user_word])
                print(selectedPin)
                firbaseService.addNewGesture(user_word, selectedPin)
                user_word = ''
                st.button("Done", on_click=enable,
                          use_container_width=True)

        else:
            enable()


# Video Page
if app_mode == "Video":
    enable()
    check_file = os.path.isfile(mainModelPath)
    if len(getGestureNames()) != 0 and check_file:
        checked = st.checkbox('automation on')
        main('', 0, checked)
    else:
        st.title("No gesture found")
