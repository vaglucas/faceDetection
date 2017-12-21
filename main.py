import cv2
import numpy as np

def main():
    facedetecitonFronVideo()

def getImage(x1,y1,x2,y2,img):
    return img[x1:y1,x2:y2]


def facedeteciton():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('eu0.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("eu",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def facedetecitonFronVideo():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            print "faces: ",faces
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            frame[x:y,x+w:y+h] = faceCane
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                print "eyes: ",eyes
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap.destroyAllWindows()


def facedetecitonFronVideo():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    cane = cv2.imread("cao.jpg",1)
    faceCane = getImage(100,180,100,180,cane);
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1,5)
        for (x,y,w,h) in faces:
            print "faces: ",faces
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            res = cv2.resize(faceCane,(w,h), interpolation = cv2.INTER_CUBIC)
            frame[y:y+h, x:x+w] = res
            eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
            for (ex,ey,ew,eh) in eyes:
                print "eyes: ",eyes
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            smile = smile_cascade.detectMultiScale(roi_gray,1.3,10)
            for (ex,ey,ew,eh) in smile:
                print "smile: ",smile
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

        cv2.imshow("gray",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap.destroyAllWindows()




if __name__ == '__main__':
    main()
