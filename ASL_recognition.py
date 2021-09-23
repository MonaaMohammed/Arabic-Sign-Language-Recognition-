from kivy.core.image import ImageLoader
from kivy.lang.builder import Builder
from kivymd.uix.screen import MDScreen
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivymd.uix.button import MDFillRoundFlatIconButton, MDFillRoundFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDToolbar
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with open('signs15.pkl', 'rb') as f:
    model = pickle.load(f)
    

def display_text_in_arabic(image,text,position=(15,4)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text) 
    img_pil = Image.fromarray(image)
    fontpath = "arial.ttf" # <== https://www.freefontspro.com/14454/arial.ttf  
    font = ImageFont.truetype(fontpath, 30)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position,bidi_text,font=font,align="right")
    img = np.array(img_pil)    
    return img

def isInitialized(hand):
    try:
        if hand.IsInitialized() == True:
            return True
    except: return False

def convertToArabic(sign):
    switcher={
        "wahed":"واحد",
        "etnyn":"اثنين",
        "tlata":"ثلاثة",
        "arb3a":"اربعة",
        "5msa":"خمسه",
        "sta":"ستة",
        "sb3a":"سبعة",
        "tmnya":"ثمانية",
        "ts3a":"تسعة",
        "34ra":"عشرة",
        "kwys":"كويس",
        "4krn":"شكرا",
        "2sf":"اسف",
        "nt3rf":"نتعرف",
        "el3nwan":"العنوان",
        "fady":"فاضى",
        "ahlnwshln":"اهلا و سهلا",
        "7adr":"حاضر",
        "esmk":"اسمك",
        "25bark":"اخبارك",
    }
    return switcher.get(sign, "nothing")


class ConverterApp(MDApp):
    
    def convert(self, args):
        
        cap = cv2.VideoCapture(0)
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
            

                # Extract left_hand landmarks
                if isInitialized(results.left_hand_landmarks) == False or isInitialized(results.right_hand_landmarks)==False or isInitialized(results.pose_landmarks)==False:
                    word = "يدك غير واضحة من فضلك تحرك خطوة للخلف او للامام لاظهارها"
                    img = display_text_in_arabic(image , word , (0,200))
                    image = img

                

                if(isInitialized(results.right_hand_landmarks)==True and  isInitialized(results.left_hand_landmarks) == True and isInitialized(results.pose_landmarks)==True):
                    # Concate rows
                    print("capturing")
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    #left hand coordinates
                    left_hand = results.left_hand_landmarks.landmark
                    left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                    #right hand coordinates
                    right_hand = results.right_hand_landmarks.landmark
                    right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

                    rowRLH = pose_row+right_hand_row+left_hand_row
                    # Make Detections
                    X = pd.DataFrame([rowRLH])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class, body_language_prob) 
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (200, 80), (245, 117, 16), -1)

                    # Display Class
                    word = "الكلمة : "
                    img = display_text_in_arabic(image , word , (120,9))

                    sign = convertToArabic(str(body_language_class.split(' ')[0]))
                    img = display_text_in_arabic(img , sign , (10,9))

                    accuracy = "النسبة : "
                    image = display_text_in_arabic(img ,accuracy ,(120,40))
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)) 
                                            , (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)




                cv2.imshow('Raw Webcam Feed', image)


                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return True




   
    def build(self):
   
        screen = MDScreen()

        # top toolbar
        self.toolbar = MDToolbar(title="Arabic Sign Language Recognition")
        self.toolbar.pos_hint = {"top": 1}
    
        screen.add_widget(self.toolbar)

          
        #screen.add_widget(Image(
         #   source="logo.png",
          #  pos_hint = {"center_x": 0.5, "center_y":0.7},
           # ))        
     
        # "CONVERT" button
        screen.add_widget(MDFillRoundFlatButton(
            text="Start",
            font_size = 17,
            pos_hint = {"center_x": 0.5, "center_y":0.5},
            on_press = self.convert
        ))

        return screen

if __name__ == '__main__':
    ConverterApp().run()
