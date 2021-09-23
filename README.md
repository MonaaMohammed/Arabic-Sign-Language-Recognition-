# J6_Hackathon_Team_Name
Team Name : ProDov

# The Team
### Team Members
* Team Leader :   Ahmed Mohamed Ahmed 
* Team Member 1 : Mona Mohamed Abdelrhman https://github.com/MonaaMohammed/Arabic-Sign-Language-Recognition-/blob/main/Picture2.jpg
* Team Member 2 : Ahmed Mohamed Ahmed https://github.com/MonaaMohammed/Arabic-Sign-Language-Recognition-/blob/main/Picture1.jpg


# Toy Project
## Problem Statement
Define which problem you solved with your project.
Automatic Number plate Recognition and Detection .
The main focus in this project is to experiment deeply with, and find
alternative solutions to the image segmentation and character recognition problems
within the License Plate Recognition framework. Three main stages are identified in
such applications. 
First, it is necessary to locate and extract the license plate region from a larger
scene image. Second, having a license plate region to work with, the alphanumeric
characters in the plate need to be extracted from the background. Third, deliver
them to an OCR system for recognition.

## Learning Process
Decribe which tools you used within the learning phase and how you achieved to build your project. Additionally, formulate what you have learning during this project.
I used python , jupyter notebook , OpenCV  and tensorflow 
I learned how to detect license plate from images and in real time from video and how to apply OCR to license  plate to extreact the plate number also I learned how to save license plates detected for future analysis and searching


 
# Coding Competition
## Problem Statement 
There have been several advancements in technology and a lot of research has been done to help the people who are deaf and dumb. Aiding the cause, Deep learning, and computer vision can be used too to make an impact on this cause.
Arabic Sign Language Recognation can be very helpful for the deaf and dumb people in communicating with others as knowing sign language is not something that is common to all.
The aim of the Arabic sign language recogniation is to provide an accurate and convenient mechanism to trascibe the signs into meaningful text or speech that communication between deaf and hearning society can easily be made.  
## Solution
Explain the following in detail:
* Steps taken for solving the problem.

1 we started to collect our own data as keypoints by using holistic model to detect Face, Hand and Pose Landmarks
2 save the keypoints in CSV file to make the train on it 
3 we Trained the data on four models to check the best one and take it to make predictions
* Frameworks/Tools/Technologies stacks used.
 Python 
 Scikit-learn 
 OpenCV
 mediapipe 
 Pandas 
* Give insights into why you chose the particular technology. Therefore, please elaborate on the assumptions you made and explain the constraints of the used technology
we used Scikit-learn library beacauce it contains  a lot of efficient  tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction and we used it for evaluation matrix as well as to leverage a trainig and testing split and it helps us to fit four models to make train and choose the best result of these models to make predictions on it. 
we used OpenCv to acually  access to webcam and extract  key points 
MediaPipe Holistic pipeline integrates separate models for pose, face and hand component .This model is responsible for extracting the landmarks from the body’s Poses and hands’ poses .
The body’s poses contains 33 landmarks and each hand contains 21 landmarks .
Each landmark has ( x , y , z , v ) coordinates. 
you can take a look how it looks like -->  https://google.github.io/mediapipe/solutions/holistic.html
## Methodology 
Explain the following in detail:
  * Explain the architecture of your project. To visualise your architecture, you might provide a flow chart, wireframes or any other graphical representation. 

  * Explain which datasets you used in the project. 
   we didn't find any Aribac sign language so we collect the data by ourselves 
   Here demo for collecting the data  https://drive.google.com/file/d/1QpqCkr2aTOt8ykII4D4TYs5ahNcCWeir/view?usp=sharing
  * Explain which machine learning models you used.
   We use RandomForestClassifier model to classify the sign .
  Although we trained four models :  
	1-LogisticRegression  (lr)
	2-RidgeClassifier (rc)
 	3-GradientBoostingClassifier (gb) 
	4-RandomForestClassifier  (rf)
 we chose the RandomForestClassifier because it has the most accuracy 
 It uses the landmarks ( records consists of only digits ) Come From the Holistics model
 Then compare it with the collected data to predict the sign then return the result .  
## weights of the trained model
https://drive.google.com/file/d/1oPvJpFdxYPBFnj1dSjuib4-MQFkhIGBy/view?usp=sharing

## Demo Video
* Attach the link of the working prototype video here in the markdown.
this link of live demo
"https://drive.google.com/file/d/1vXQqmfkNBL8b96hZ6zv-8D2892Pexeur/view?usp=sharing"
## Steps of setup the software 
## 1. clone this repository https://github.com/MonaaMohammed/Arabic-Sign-Language-Recognition-/blob/main/signs_detection.ipynb
## 2.Create a new virtual environment 
python -m venv aslr
## 3.Activate your virtual environment
.\aslr\Scripts\activate # Windows
## Step 4. Install dependencies and add virtual environment to the Python Kernel
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=aslr 
## Step 5. Open jupyet-Notebook  -ensure you change the kernel to the virtual environment 
https://github.com/MonaaMohammed/Arabic-Sign-Language-Recognition-/blob/main/WhatsApp%20Image%202021-09-23%20at%205.09.46%20PM.jpeg

