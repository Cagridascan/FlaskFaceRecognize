import cv2, numpy, sys, os,json
import mediapipe as mp


def savemodel(dir_path):
    #iniilazing objects
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness= 1, circle_radius= 1)

    haarFile = "haarcascade_frontalface_default.xml"
    datasets = dir_path

    #create a list of images and a list of corresponding names
    (images , labels, names, id) = ([], [], {}, 0)
    labelmap = {}
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir

            print(id,subdir)

            labelmap.update({id:subdir})

            subjectpath = os.path.join(datasets, subdir)

            for filename in os.listdir(subjectpath):
                path = subjectpath + "/" + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))

            id += 1
    with open('labelmap.json', 'w') as jsonfile:
        json.dump(labelmap, jsonfile, indent=4)


savemodel(r"datasets")

