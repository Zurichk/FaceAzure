import os
from flask import Flask, render_template, request
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import Emotion
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import requests

app = Flask(__name__)

imagenAExaminar = 'https://images.ecestaticos.com/PJP0ljYk4EXHhNfX2xPl2veEb8M=/0x19:619x367/557x418/filters:fill(white):format(jpg)/f.elconfidencial.com%2Foriginal%2F305%2F365%2F960%2F30536596002888fe41f8a1a574a16a68.jpg'
# Cree un FaceClient autenticado. (API)
ENDPOINT = 'https://iapractica.cognitiveservices.azure.com'
KEY = "267e8744e69842fdbd11a8f8c60bfa4c"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'true',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion'
}


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/process', methods=['POST'])
def procesa_imagen():
    # Detectar un rostro en una imagen que contiene un solo rostro
    # single_face_image_url = imagenAExaminar
    if request.method == 'POST':
        urlimage = request.form['url']
        nombreimagen = os.path.basename(urlimage)
        f = open('imagenprueba.jpg', 'wb')
        descargaimagen = requests.get(urlimage)
        f.write(descargaimagen.content)
        f.close()
        imagenpath = os.path.join(
            os.path.dirname(__file__), 'imagenprueba.jpg')
        image_data = open(imagenpath, 'rb')  # .read()
        faces = face_client.face.detect_with_stream(image_data, params)
        img = Image.open(BytesIO(descargaimagen.content))

        emociones = requests.post(
            ENDPOINT, params=params, data=image_data)
        print(urlimage)
        single_image_name = os.path.basename(urlimage)
        # Utilizamos el modelo de detección 3 para obtener un mejor rendimiento.
        detected_faces = face_client.face.detect_with_url(
            url=urlimage, detection_model='detection_03')
        # print(detected_faces)

        # Verificamos que se haya detectado un rostro
        if not detected_faces:
            raise Exception(
                'No se ha detectado ninguna cara {}'.format(single_image_name))

        # Muestra la identificación de la cara detectada en la primera imagen de una sola cara.
        #  Los ID de rostros se utilizan para compararlos con los rostros (sus ID) detectados en otras imágenes.
        print('Identificación facial detectada de', single_image_name, ':')
        numerodecaras = len(detected_faces)
        for face in detected_faces:
            print(face.face_id)
            # Obtenemos la identificación de la cara
            face_ids = [face.face_id for face in detected_faces]
            face_emotions = [face.face_emotion_id for face in detected_faces]

        return render_template('results.html',  imagen=face_ids, nombre=single_image_name, numerodecaras=numerodecaras, img=img, emociones=face_emotions)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
