import firebase_admin
from firebase_admin import credentials, firestore
import threading
from time import sleep
import RPi.GPIO as GPIO

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


gesturePinDB = db.collection('automation').document('gesture_pin')
onOff = db.collection('automation').document('on_off')
GPIO.setmode(GPIO.BOARD)
boolValue = False


def getGesturePinDetails():
    data = onOff.get()
    return data.to_dict()


# event for notifieng main thread
callback_done = threading.Event()

# Callback on snapshot function to capture changes


def on_snapshot(doc_snapshot, changes, read_time):
    gestPin = getGesturePinDetails()
    for pin in gestPin:
        for doc in doc_snapshot:
            docDict = doc.to_dict()
            isTrue = docDict[pin] == False
            GPIO.setup(int(pin), GPIO.OUT)
            GPIO.output(int(pin), isTrue)
            sleep(0.1)
            global boolValue
            boolValue = docDict[pin]
            print(f"{pin} = {boolValue}")
        callback_done.set()


doc_watch = onOff.on_snapshot(on_snapshot)

while True:
    sleep(0.1)
