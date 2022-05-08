import os
from flask import Flask, render_template, request
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
# Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import time

app = Flask(__name__)

# Cree un FaceClient autenticado. (API)
ENDPOINT = 'https://iapractica.cognitiveservices.azure.com'
KEY = "267e8744e69842fdbd11a8f8c60bfa4c"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Computer Vision
domain = "landmarks"
region = 'eastus'
key = '040373193c5746ee8a371e93d9c98190'
credentials = CognitiveServicesCredentials(key)
cliente = ComputerVisionClient(
    endpoint="https://" + region + ".api.cognitive.microsoft.com/",
    credentials=credentials)


def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    return ((rect.width, rect.height), (left, top))


def get_emotion(emoObject):
    emoDict = dict()
    emoDict['Ira'] = emoObject.anger
    emoDict['Desprecio'] = emoObject.contempt
    emoDict['Asco'] = emoObject.disgust
    emoDict['Miedo'] = emoObject.fear
    emoDict['Felicidad'] = emoObject.happiness
    emoDict['Neutral'] = emoObject.neutral
    emoDict['Tristeza'] = emoObject.sadness
    emoDict['Sorpresa'] = emoObject.surprise
    emo_name = max(emoDict, key=emoDict.get)
    emo_level = emoDict[emo_name]
    return emo_name, emo_level


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/process', methods=['POST'])
def procesa_imagen():
    imagen = ""
    id_cara = ""
    # Detectar un rostro en una imagen que contiene un solo rostro
    if request.method == 'POST':
        urlimage = request.form['url']
        nombreimagen = os.path.basename(urlimage)

        face_attributes = ['emotion', 'age']
        response_caras_detectadas = face_client.face.detect_with_url(
            urlimage, detection_model='detection_01', recognition_model='recognition_01', return_face_attributes=face_attributes)
        # Verificamos que se haya detectado un rostro ----> Comento estas lineas por que tambien podemos incluir imagenes sin caras para convertir imagen a texto
        """if not response_caras_detectadas:
            raise Exception(
                'No se ha detectado ninguna cara {}'.format(response_caras_detectadas))"""

        numerodecaras = len(response_caras_detectadas)
        personas = []

        for face in response_caras_detectadas:
            id_cara = face.face_id
            #edad_cara = face.face_attributes.age(face, attributes='age')
            edad_cara = face.face_attributes.age

            emotion, confidence = get_emotion(face.face_attributes.emotion)
            posderechaabajo, posizquierdaarriba = getRectangle(face)
            personas.append("ID Cara: " + str(id_cara))
            personas.append("Posicion: " + "(Ancho,Alto)" + str(posderechaabajo) + " (Izquierda,Arriba)" +
                            str(posizquierdaarriba))
            personas.append("Edad: " + str(int(edad_cara)) + " años")
            personas.append("Emocion: " + str(emotion) +
                            " (Confianza: " + str(confidence) + " de 1.0)")
            personas.append("--------------")

        # -------------------------- Analizando la imagen --------------------------------
        features = ["color",
                    "description", "tags"]
        details = ["landmarks"]
        resultado1 = cliente.analyze_image(
            urlimage, features, details, 'en')

        accentcolor = resultado1.color.accent_color

        # Descripción de la imagen
        # for caption in resultado1.description:
        #print(caption.text, caption.confidence * 100)
        # Obtener la descripción de texto de una imagen
        language = "es"
        max_descriptions = 1
        analisis = cliente.describe_image(urlimage, max_descriptions, language)
        analizado = []
        for caption in analisis.captions:
            analizado.append("Descripción: " + str(caption.text))
            confianza = (caption.confidence * 100)
            analizado.append(
                "Confianza: " + str(round(confianza, 2)) + "%")
            analizado.append("--------------")

        # Etiquetas de la imagen
        # Forma 1
        etiquetas = []
        image_analysis = cliente.analyze_image(
            urlimage, visual_features=[VisualFeatureTypes.tags])
        for tag in image_analysis.tags:
            etiquetas.append("Etiqueta: " + str(tag))
        etiquetas.append("--------------")
        # Forma 2 ->> Usada en html
        tags = []
        for tag in resultado1.tags:
            confianzatags = (tag.confidence * 100)
            tags.append("Etiqueta: " + str(tag.name) + " " +
                        str(round(confianzatags, 2)))
        tags.append("--------------")

        # Obtener texto de la imagen
        textoencontrado = []
        numberOfCharsInOperationId = 36
        rawHttpResponse = cliente.read(
            urlimage, language='en', raw=True)
        operationLocation = rawHttpResponse.headers["Operation-Location"]
        idLocation = len(operationLocation) - numberOfCharsInOperationId
        operationId = operationLocation[idLocation:]
        resultado = cliente.get_read_result(operationId)

        while resultado.status != OperationStatusCodes.succeeded and resultado.status != OperationStatusCodes.failed:
            resultado = cliente.get_read_result(operationId)
            time.sleep(1)

        if resultado.status == OperationStatusCodes.succeeded:  # OperationStatusCodes.running
            for r in resultado.analyze_result.read_results[0].lines:
                print(r.text)
                textoencontrado.append(r.text)
            textoencontrado.append("--------------")

        if imagen == None:
            imagen = "No se ha encontrado ninguna cara"

        return render_template('results.html', imagen=id_cara, personas=personas, nombre=nombreimagen,
                               numerodecaras=numerodecaras, analizado=analizado, textoencontrado=textoencontrado,
                               etiquetas=etiquetas, accentcolor=accentcolor, tags=tags)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
