import cv2
from keras.models import load_model
import numpy as np
from skimage import transform
import os


class FaceTracker:

    def __init__(self, bbox_predictor_path='', face_detector_path='', is_coords_normalized=True, is_img_gryscl=False, resize=(128,128), frame_height=640, frame_width=480):

        self.bbox_predictor_path = bbox_predictor_path
        self.face_detector_path = face_detector_path
        self.is_coords_normalized = is_coords_normalized
        self.is_img_gryscl = is_img_gryscl
        self.resize = resize
        self.frame_height = frame_height
        self.frame_width = frame_width
    
    def _preprocess_frame(self, frame):

        """
        Transforms Frame by resizing and/or grayscaling

        Arugments:
            
            frame: numpy array of shape: (image_width, image_height, num_channels)
        
        Returns:

            preprocessed frame of shape: (1, resize, resize, 1 or 3 channels)
        """

        dims = (self.resize[0], self.resize[1]) if self.resize else (self.frame_height, self.frame_width)

        frame = np.array(frame)
        frame_resized = transform.resize(frame, dims, mode='reflect', anti_aliasing=True) *255
        frame_resized = frame_resized.astype(np.uint8)

        if self.is_img_gryscl:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            frame_resized = np.reshape(frame_resized, (1, dims[0], dims[1], 1))
        else:
            frame_resized = np.reshape(frame_resized, (1, dims[0], dims[1], 3))
        return frame_resized
    
    def get_face_rect_coords(self, frame, model):

        """
        Fetches rectangle coordinates from bboxes estimator given a frame; Normalizes bboxes if is_coords_normalized == True

        Arugments:

            frame: numpy array of shape: (image_width, image_height, num_channels)

            model: bboxes predictor estimator
        
        Returns:

            bboxes like the following: [0, 0, 300, 480] => [xmin, ymin, xmax, ymax]
        """

        if os.path.exists(self.bbox_predictor_path):

            frame_preproc = self._preprocess_frame(frame)

            bboxes = model.predict(frame_preproc, verbose=0)[0]

            if self.is_coords_normalized:

                coords_unnormalized = []
                for idx, x in enumerate(bboxes):
                    if idx % 2 == 0:
                        x = int(abs(x)*self.cam_dims[1])
                        coords_unnormalized.append(x)
                    else:
                        x = int(abs(x)*self.cam_dims[0])
                        coords_unnormalized.append(x)
                bboxes = coords_unnormalized
                
            return bboxes
        else:
            print("Path to BBox predictor is broken: ",self.bbox_predictor_path)
            return
    
    def is_face_in_frame(self, frame, model):

        """
        Determines whether a face is present within a frame

        Arguments:

            frame: numpy array of shape: (image_width, image_height, num_channels)

            model: face detector estimator
        
        Returns:

            boolean (True if face detected, False if no face detected)
        """

        if os.path.exists(self.face_detector_path):

            frame_preproc = self._preprocess_frame(frame)

            pred = model.predict(frame_preproc, verbose=0)[0]

            pred = np.round(pred, 0)[0]

            return bool(pred)
        
        else:
            print("Path to face detector is broken: ",self.face_detector_path)
    
    def run(self):

        """
        Main application loop. Opens default video recording device and passes frames to each estimator for predictions

        Arguments:

            None
        
        Returns:

            -1 if error occured during initialization of recording device.
        """


        if os.path.exists(self.bbox_predictor_path) and os.path.exists(self.face_detector_path):

            
            try:
                vid = cv2.VideoCapture(0)
                if not vid.isOpened():
                    print("Please ensure you have a webcam connected.")
                    return -1
                
            except Exception as e:
                print("An unexpected error occurred while opening the webcam:", e)
                return -1

            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_dims[0])
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_dims[1])

            shooting = True

            face_dect = load_model(self.face_detector_path)
            bbox_pred = load_model(self.bbox_predictor_path)

            while shooting:

                ret, frame = vid.read()

                if ret and vid.isOpened():

                    is_face_detected = self.is_face_in_frame(frame, face_dect)
                    
                    if is_face_detected:

                        bboxes = self.get_face_rect_coords(frame, bbox_pred)
                        print("Face detected at coords: ",bboxes)

                        cv2.rectangle(frame, pt1=(bboxes[0], bboxes[1]), pt2=(bboxes[2], bboxes[3]), color=(255, 0, 0), thickness=3)


                    cv2.imshow('Face detector programmed by Franklin Collazo', frame)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shooting = False

            vid.release()
            cv2.destroyAllWindows()
            
        else:
            print("Paths to either face detector or bbox predictor is broken.")


if __name__ == "__main__":
    print("File is meant for importing, not running.")