#Creating Data from jpg files


#importing the libraries
import cv2, sys, numpy, os

from save_model import savemodel


#address of the haar file
haarFile = "haarcascade_frontalface_default.xml"


#datasets file directory
datasets = "datasets"

#for data directory name
sub_data = input("Enter the name of the data:")

#create and check the file path
path = os.path.join(datasets, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)

#defining the weight and height
(width, height) = (130, 100)


face_cascade = cv2.CascadeClassifier(haarFile)
#


directory = input("enter images directory path:")
count = 1

for root, dirs, files in os.walk(directory):
    for name in files:
        image= os.path.join(root, name)

        image= cv2.imread(image)

        #convert the color space from BGR to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:

            cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),2)
            face = gray[y:y + h, x:x+w]

            face_resize = cv2.resize(face, (width,height))
            cv2.imwrite("% s/% s.png" % (path,count),face_resize)

        count += 1

        # cv2.imshow("CreateData",image)

#update datasets using save_model.py
savemodel(datasets)
print("well done")