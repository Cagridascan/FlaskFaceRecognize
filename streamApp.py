#Face Recognizing and Face Landmarks Stream

import cv2, numpy, sys, os
import mediapipe as mp

from flask import Flask, render_template, Response


app = Flask(__name__)


def generate_frames():
    #Initilazing Objects
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness= 1, circle_radius= 1)
    webcam = cv2.VideoCapture(0)

    #defining for face recognize

    haarFile = "haarcascade_frontalface_default.xml"
    datasets = "datasets"

    print("Recognizing Face Please Be in Sufficient Light")

    #create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):

        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)

            for filename in os.listdir(subjectpath):
                path = subjectpath + "/" + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))


            id += 1

    (width , height)=(130,100)

    #Create a Numpy array from the two lists above
    (images , labels) = [numpy.array(lis) for lis in [images , labels]]

    #OpenCV trains a model from the images
    #NOTE FOR OpenCV': remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

    face_cascade = cv2.CascadeClassifier(haarFile)
    success, image = webcam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count = len(faces)

    with mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_faces=face_count) as face_mesh:
        while True:


            success , image = webcam.read()
            gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3 , 5)

            #convert the color space from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #To improve performance
            image.flags.writeable = False

            #Detect the face landmarks
            results = face_mesh.process(image)

            #To improve performance
            image.flags.writeable = True

            #convert back to the BGR color
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Draw the face mesh annotations on the image
            if results.multi_face_landmarks:
                for face_landmmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image = image,
                        landmark_list= face_landmmarks,
                        connections= mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec= None,
                        connection_drawing_spec= mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face,(width, height))

                #Try to recognize the face
                prediction = model.predict(face_resize)
                cv2.rectangle(image, (x,y),(x+w, y+h), 3)

                if prediction[1]/5<100 and prediction[1] <= 85:

                    cv2.putText(image,"% s - %.0f"%
                            (names[prediction[0]], prediction[1]), (x-10, y-10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3 , (0,255,0)

                           )


                elif prediction[1] < 500 and prediction[1] >= 85:
                    cv2.putText(image, "not recognized", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))


                else:
                    cv2.putText(image, "not recognized",
                                (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)

                                )

            # display on the web browser
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')



@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)