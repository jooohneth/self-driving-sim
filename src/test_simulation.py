import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LegacyCompatServer(socketio.Server):
    """auto-connects legacy socket.io v1/v2 clients that send event packets
    without first sending a namespace connect packet."""

    def _handle_eio_message(self, sid, data):
        if isinstance(data, str) and data.startswith('2['):
            namespace = '/'
            sio_sid = self.manager.sid_from_eio_sid(sid, namespace)
            if sio_sid is None:
                self._handle_connect(sid, namespace, None)
        super()._handle_eio_message(sid, data)


sio = LegacyCompatServer(async_mode='threading', cors_allowed_origins='*')
app = Flask(__name__)
maxSpeed = 9


def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print(f'{throttle}, {steering}, {speed}')
    sio.emit('steer', {
        'steering_angle': str(steering),
        'throttle': str(throttle)
    }, room=sid)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sio.emit('steer', {'steering_angle': '0', 'throttle': '0'}, room=sid)


if __name__ == '__main__':
    model = load_model(str(PROJECT_ROOT / 'models' / 'model.h5'))
    flask_app = socketio.WSGIApp(sio, app)
    import werkzeug.serving
    werkzeug.serving.run_simple('0.0.0.0', 4567, flask_app, threaded=True)
