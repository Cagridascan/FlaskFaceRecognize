import cv2, numpy, sys, os, json
import mediapipe as mp


def Predict(image_path):
    # iniilazing objects
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    haarFile = "haarcascade_frontalface_default.xml"
    datasets = "datasets"

    # image_path = r"C:\Users\cagri\PycharmProjects\faceDetection\images\cagri.jpg"
    image_cam = cv2.imread(image_path)

    # create a list of images and a list of corresponding names
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
                # end of for
            id += 1
            # end of for
        # end of for
    (width, height) = (130, 100)

    # json file operations
    fdata = open("labelmap.json")
    jdata = json.load(fdata)
    fdata.close()

    # create a numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    # opencv trains a model from the images
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    # using haarcascade classifier
    face_cascade = cv2.CascadeClassifier(haarFile)


    image = image_cam
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for landmark while image has multi faces
    face_count = len(faces)

    # convert the color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                               max_num_faces=face_count) as face_mesh:

        # to improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results = face_mesh.process(image)

        # to improve performance
        image.flags.writeable = True

        # convert back to th BGR color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the face mesh annotations on the image
        if results.multi_face_landmarks:
            for face_lanmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_lanmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            # Try to recognize the face
            prediction = model.predict(face_resize)

            cv2.rectangle(image, (x, y), (x + w, y + h), 3)

            # names[prediction[0]]
            if prediction[1] / 500 < 1 and prediction[1] < 85:
                cv2.putText(image, "% s - %.0f" %
                            (jdata[str(int(prediction[0]))], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0))

            elif prediction[1] < 500 and prediction[1] >= 85:
                cv2.putText(image, "not recognized", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            else:
                cv2.putText(image, "not recognized", (x - 10, y - 10), cv2.FONT_HERSHEY, 1, (0, 255, 0))

            # end of for

    return image


