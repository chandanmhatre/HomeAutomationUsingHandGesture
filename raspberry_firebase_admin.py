import firebase_admin
from firebase_admin import credentials, firestore
import threading
from time import sleep


if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


gesturePinDB = db.collection('automation').document('gesture_pin')
onOff = db.collection('automation').document('on_off')
plugPin = db.collection('automation').document('plug_pin')

if db.collection('automation').get() == []:
    plugPin.set({
        '1st': 35,
        '2nd': 36,
        '3rd': 37,
        '4rth': 38
    })
    onOff.set({
        '35': False,
        '36': False,
        '37': False,
        '38': False
    })
    gesturePinDB.set({})


class FirbaseServices(object):
    def addNewGesture(self, gestureName, pinNo):
        # if db.collection('automation').get() == []:
        #     gesturePinDB.set({
        #         gestureName: pinNo
        #     })
        #     onOff.set({
        #         str(pinNo): False
        #     })
        # else:
        gesturePinDB.update({
            gestureName: pinNo
        })
        onOff.update({
            str(pinNo): False
        })

    def updatePinValue(self, pinNo, onOrOff):
        onOff.update({
            str(pinNo): onOrOff
        })

    def deleteField(self, gestureName):
        gesturePinDB.update({
            gestureName:  firestore.DELETE_FIELD
        })
        # onOff.update({
        #     str(pinNo):  firestore.DELETE_FIELD
        # })

    def getGesturePinDetails(self):
        data = onOff.get()
        return data.to_dict()

    def getPlugPinDetails(self):
        data = plugPin.get()
        return data.to_dict()
