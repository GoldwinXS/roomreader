import cv2,os
import matplotlib.pyplot as plt
import face_recognition.face_recognition as fr
import numpy as np

# test_file = 'test4.jpg'
# pixels = plt.imread(test_file)


def detect_faces(image):

    """ simply draw a rectangle around the face in question """

    faces = fr.face_locations(image)

    for face in faces:
        image = cv2.rectangle(image,
                              (face[1], face[0]),
                              (face[3], face[2]),
                              color=(0, 255, 0),
                              thickness=2)

    return image


def detect_face_emotions(image,predictions):

    """ simply draw a rectangle around the face in question """

    faces = fr.face_locations(image)

    for i,face in enumerate(faces):
        image = cv2.rectangle(image,
                              (face[1], face[0]),
                              (face[3], face[2]),
                              color=(0, 255, 0),
                              thickness=2)

        image = cv2.putText(image,text=predictions[i],org=(face[3],face[0]),color=(255,255,255),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2)

    return image

def extract_faces(image):

    """ get all faces from an image. Returns as a list """

    faces = fr.face_locations(image)
    imgs = []

    for face in faces:
        x = face[3]
        y = face[0]

        wx = face[1]
        wy = face[2]

        imgs.append(image[y:wy, x:wx])

    return imgs


def prepare_cropped_images(images, size=(48, 48)):

    """ take image of a face and crop/resize so that they are compatible with the emotion model """

    prepared_images = []

    for image in images:

        wx = image.shape[1]
        wy = image.shape[0]

        if wx > wy:
            image = image[:, :wy]

        elif wy > wx:
            image = image[:wx, :]

        image = cv2.resize(image, size)
        prepared_images.append(image)

    return prepared_images




class image_emotion_dectector:
    def __init__(self):
        from keras.models import load_model
        self.model = load_model('emotionCNN.h5')
        self.category_names = os.listdir('emotions/')
        self.category_names = [x for x in self.category_names if x != '.DS_Store']

    def predict(self, image):
        faces = np.array(prepare_cropped_images(extract_faces(image)))
        str_preditions = []

        if len(faces.shape)>1:
            predictions = self.model.predict(faces)

            for prediction in predictions:
                str_preditions.append(self.num_to_emostr(np.argmax(prediction)))

        return str_preditions

    def num_to_emostr(self,prediction):
        emostr = self.category_names[prediction]
        return emostr


def find_faces_in_video(ds_factor=2):

    """ draws a rectangle around any faces in the frame """

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    downsample_factor = ds_factor

    rval, frame = vc.read()

    while True:

        if frame is not None:
            cv2.imshow("preview", detect_faces(frame))
        rval, frame = vc.read()

        w = int(frame.shape[1] / downsample_factor)
        h = int(frame.shape[0] / downsample_factor)

        frame = cv2.resize(frame, (w, h))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# find_faces_in_video(ds_factor=1)

