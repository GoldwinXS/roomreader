from ProjectUtils import *
import cv2


emotion_detector = image_emotion_dectector()
# emotion_detector.predict(prepare_cropped_images(cropped_images[0]))


def show_face_emotions(ds_factor=2):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    downsample_factor = ds_factor
    rval, frame = vc.read()

    while True:

        if frame is not None:
            cv2.imshow("preview", frame)
        rval, frame = vc.read()

        predictions = emotion_detector.predict(frame)
        frame = detect_face_emotions(frame, predictions)


        w = int(frame.shape[1] / downsample_factor)
        h = int(frame.shape[0] / downsample_factor)

        frame = cv2.resize(frame, (w, h))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


show_face_emotions(ds_factor=2)