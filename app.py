import csv
import os
import pickle
import time
import cv2
import pandas as pd
from utils.landmark_servcies import landMarkServices
from utils.raspberry_firebase_admin import FirbaseServices
import streamlit as st

from utils.utils import Utils


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
                brect = landMarkService.calc_bounding_rect(
                    frame, handLandMarks)
                global landmark_list
                global handSide
                handSide = handedness.classification[0].label[0:]
                landmark_list = landMarkService.calc_landmark_list(
                    frame, handLandMarks)
                myHand = []
                for landMark in handLandMarks.landmark:
                    myHand.append(
                        (int(landMark.x*width), int(landMark.y*height)))
                myHands.append(myHand)
        return myHands


mainModelPath = 'dataset/MainModel/main.pkl'
newGestureSavingPath = 'dataset/model/'
gesturePlugPinCsvPath = 'dataset/gesture-plug-pin.csv'
detectionKeypointItteration = 10
firbaseService = FirbaseServices()
landMarkService = landMarkServices()
findHands = mpHands()
utils = Utils()


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
        keyPoints = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
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
        cancelBtnHolder.empty()
        with st.empty():
            while checkIsTestingOrTraining(isTraining, knownGestures):
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (width, height))
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                handData = findHands.Marks(frame)
                if train == 1:
                    buildTrainingFrame(knownGestures, frame, handData)
                if train == 0:
                    frame = buildTestingFrame(
                        automationOn, keyPoints, knownGestures, gestNames, tol, frame, handData)
                st.image(frame, use_column_width=True)
            if train == 1:
                with open(newGestureSavingPath+trainModelName, 'wb') as f:
                    pickle.dump(gestNames, f)
                    pickle.dump(knownGestures, f)
                    # logging_keypoints_csv(gestNames,  knownGestures)
                utils.buildMainModel()
                global isLoggingComplete
                isLoggingComplete = True


def buildTrainingFrame(knownGestures, frame, handData):
    cv2.putText(
        frame,
        "Remaining : " +
        str(detectionKeypointItteration - len(knownGestures)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )
    if handData != []:
        time.sleep(1)
        knownGesture = utils.findDistances(handData[0])
        knownGestures.append(knownGesture)


def buildTestingFrame(automationOn, keyPoints, knownGestures, gestNames, tol, frame, handData):
    if handData != []:
        unknownGesture = utils.findDistances(handData[0])
        detectedGesture = utils.findGesture(
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
            gesturepinDataList = getGesturePlugPin()
            for gesturePin in gesturepinDataList:
                # gesture = 0, plug = 1, pin = 2
                serverOnOff = firbaseService.getGesturePinDetails()
                if gesturePin[0] == detectedGesture:
                    selectedPin = gesturePin[2]
                    onOrOff = True if serverOnOff[str(
                        selectedPin)] == False else False
                    firbaseService.updatePinValue(
                        str(selectedPin), onOrOff)
                    time.sleep(2)
        frame = landMarkService.draw_landmarks(
            frame, landmark_list)

    return frame


def checkIsTestingOrTraining(isTraining, knownGestures):
    if (isTraining == 0):
        return True
    else:
        return len(knownGestures) < detectionKeypointItteration


# def logging_keypoints_csv(name,  knownGestures):
#     csv_path = 'keypoint.csv'
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([name, *knownGestures])


state = st.session_state
st.markdown("<h2 style='text-align: center;'>Home automation using hand gesture</h2>",
            unsafe_allow_html=True)

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
st.sidebar.title("Home Automation")

app_mode = st.sidebar.selectbox(
    "Menu", ["Set Gesture", "View/Automate", "Config", "About"])
if "disabled" not in state:
    state["disabled"] = False


def disable():
    state["disabled"] = True


def enable():
    state["disabled"] = False


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


def getGestureNames():
    with open(gesturePlugPinCsvPath, encoding="utf-8-sig"
              ) as f:
        gesture_labels = csv.reader(f)
        gesture_labels = [row[0] for row in gesture_labels]
    return gesture_labels


def getGesturePin():
    with open(gesturePlugPinCsvPath, encoding="utf-8-sig"
              ) as f:
        gesture_pin_labels = csv.reader(f)
        gesture_pin_labels = [row[2] for row in gesture_pin_labels]
    return gesture_pin_labels


def getGesturePlugPin():
    with open(gesturePlugPinCsvPath, encoding="utf-8-sig"
              ) as f:
        gesture_pin_labels = csv.reader(f)
        gesture_pin_labels = [row for row in gesture_pin_labels]
    return gesture_pin_labels


def checkDuplicatePinAndUpdatePin(selectedPin, gestureName):
    isDuplicatePin = False
    with open(gesturePlugPinCsvPath, 'r') as read:
        file = csv.reader(read, delimiter=',')
        lines = list(file)
        lines = list(filter(None, lines))
        for obj in lines:
            if (gestureName == str(obj[0])):
                isDuplicatePin = True
                st.error(
                    f'Gesture name "{gestureName}" is already entered for plug "{getPlugNameUsingPin()}"')
                break
            if str(selectedPin) == str(obj[2]):
                isDuplicatePin = True
                st.error(
                    f'"{getPlugNameUsingPin()}" plug is already selected for gesture "{obj[0]}"')
                break
    return isDuplicatePin


def getPlugNameUsingPin():
    for pin in plugPinDict:
        if plugPinDict[pin] == selectedPin:
            return pin


def logGesturesNamePinPlug():
    with open(gesturePlugPinCsvPath, 'a', newline="", encoding="utf-8") as append:
        file = csv.writer(append)
        file.writerow([user_word, getPlugNameUsingPin(), selectedPin])


# set gesture page
cancelBtnHolder = st.empty()
if app_mode == "Set Gesture":
    global plugPinDict
    plugPinDict = firbaseService.getPlugPinDetails()
    sortedPluPinList = sorted(plugPinDict)
    isLoggingComplete = False
    with st.form(key="Form1", clear_on_submit=False):
        with st.sidebar:
            selectedPinName = st.sidebar.selectbox(
                "Select Plug", sortedPluPinList
            )
            user_word = st.sidebar.text_input(
                "Enter Gesture Name", disabled=state.disabled, on_change=disable
            )

    selectedPin = plugPinDict[selectedPinName]
    if user_word in getGestureNames():
        st.error(f'{user_word} already exit')
        enable()
        changeBtn = st.button('change name')
        if changeBtn:
            enable()
    else:
        if user_word != "" and state["disabled"] and checkDuplicatePinAndUpdatePin(selectedPin, user_word) == False:
            cancelBtnHolder.button("Cancel", on_click=enable,
                                   use_container_width=True, key='cancel')
            main(user_word, 1, False)
            if isLoggingComplete:
                logGesturesNamePinPlug()
                firbaseService.addNewGesture(user_word, selectedPin)
                st.button("Done", on_click=enable,
                          use_container_width=True)
                user_word = ''

        else:
            enable()


# Video Page
if app_mode == "View/Automate":
    enable()
    check_file = os.path.isfile(mainModelPath)
    if len(getGestureNames()) != 0 and check_file:
        cancelBtnHolder.button("Cancel", on_click=enable,
                               use_container_width=True, key='cancel')
        checked = st.checkbox('automation on')
        main('', 0, checked)
    else:
        st.title("No gesture found")


def deleteGesture(gestureName, csvPath):
    lines = []
    with open(csvPath, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            lines.append(row)
            for field in row:
                if field == gestureName:
                    lines.remove(row)
                    break
    with open(csvPath, 'w', newline="", encoding="utf-8") as writeFile:
        writer = csv.writer(writeFile)
        for name in lines:
            writer.writerow(name)


# config Page
if app_mode == "Config":
    isEmpty = False
    df = pd.read_csv(gesturePlugPinCsvPath, sep=',',
                     names=["Gesture", "Plug", 'Pin'])
    df = pd.DataFrame(df, columns=["Gesture", "Plug", 'Pin'])
    df.drop(df.iloc[:, 2:], inplace=True, axis=1)
    st.markdown(
        "<p style='text-align: center; color: orange;font-size:30px;'>---- Saved Gesture Information ----</p>", unsafe_allow_html=True)
    st.dataframe(df.style.set_properties(**{'background-color': 'white',
                                            'color': 'black'}), use_container_width=True)
    nameList = [""]
    for name in getGestureNames():
        nameList.append(name)
    st.write("####")
    st.write("####")
    st.write("####")
    st.markdown(
        "<p style='text-align: center; color: orange;font-size:30px;'>---- Delete Gesture ----</p>", unsafe_allow_html=True)
    selectedGesture = st.selectbox(
        "Select Gesture",  nameList)
    deleteBtn = st.button("Delete Gesture", use_container_width=True)
    if selectedGesture != '' and deleteBtn:
        deleteGesture(selectedGesture, gesturePlugPinCsvPath)
        firbaseService.deleteField(selectedGesture)
        os.remove(newGestureSavingPath+selectedGesture+'.pkl')
        utils.buildMainModel()


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
