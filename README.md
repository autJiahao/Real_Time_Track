# Auckland University of Technology

## Research and Development

### Real-time Traffic Video Generator

#### Primary used Skills
`OpenCv` `matplotlib` `PyQt5` `Yolov5` `Deep_Sort`

# Purpose of this project🧐

> This program provides a PyQt-based interface that uses YOLOv5 and DeepSort for object detection and tracking in videos.

> When the program is executed, the user can select a video file. The selected video is then processed using the YOLOv5 model to detect objects, and the detected objects are tracked using the DeepSort algorithm. Tracked objects are assigned unique identifiers (IDs), and their trajectories are visually displayed.

>The interface includes "Play," "Pause," and "Replay" buttons. Clicking the "Play" button starts the object detection and tracking. The "Pause" button allows pausing the detection and tracking process, and the "Replay" button restarts the detection and tracking from the beginning.

<img width="1439" alt="Screenshot 2023-05-17 at 1 14 08 PM" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/20b92c3c-78e8-4359-b100-332e6b97d476">

>A dropdown list is provided for selecting the model, allowing the user to choose from various pretrained YOLOv5 models. The selected model is used for real-time object detection and tracking.

>This program provides a user-friendly environment for visualizing object detection and tracking, making it useful for tasks in video processing and computer vision.

# 🛠️ Usage

- [Setup](#setup)
- [Information](#Information)
- [Summary](#Summary)
- [Analysis](#Analysis)

## Setup


Step 1️⃣ Clone our repository to your local machine.
<br />

    git clone https://github.com/autJiahao/Real_Time_Track.git  # clone

Step 2️⃣ Navigate to the downloaded folder and change directory to our file. 
<br />

    cd Real_Time_Track

Step️ 3️⃣ Install the dependencies listed in the requirement.txt file.
<br />
    
    pip install -r requirements.txt  # install

<br />

## Information
DEMO1,2,3,4,5

<br />

## Summary

1. Import Statements:
    - Importing required modules and classes from different Python files and libraries. 
   <br />
    <img width="997" alt="1  import" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/da31d603-10ce-4d44-a9c8-b8b9c0559085">

2. Yolov5 Thread Class:
    - A custom QThread subclass that runs YOLOv5 object detection on input frames.
    - The run method performs the main detection and annotation logic.
    - The send_img signal is emitted to send the annotated frames to the GUI.
    <br />
    <img width="1015" alt="2  yolov5 thread" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/169252ae-0848-4d65-b198-dc23a4e339ab">

3. Deepsort Thread class:
    - Another custom QThread subclass that runs DeepSort object tracking on the detected objects from YOLOv5.
    - The run method performs the tracking and annotation logic.
    - The send_img2 signal is emitted to send the annotated frames to the GUI.
    <br />
    <img width="1026" alt="3  deepsort thread" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/0584c701-ff0c-4951-a7ac-8e5d9dd8372b">

4. Window Class:
    - Inherits from QWidget and represents the main application window.
    - Sets up the GUI elements, such as buttons, labels, and file dialogs.
    - Handles button clicks and connects them to appropriate actions.
    - Initializes instances of Yolov5Thread and DeepsortThread classes.
    - Contains methods for displaying images in the GUI labels.
    <br />
    <img width="1037" alt="4  Window" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/498dbdd1-5fc3-4547-b488-d837a071142f">

5. Main Execution:
    - Initializes the QApplication and creates an instance of the Window class.
    - Shows the window and starts the application event loop.
    <br />
    <img width="1016" alt="5  main execution" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/a6354d3e-bfcc-47ff-982d-18cc3ca62a81">



## Analysis





----
헤더 #
줄바꿈 빈줄2줄
소스코드 위아래띄고 4줄인덴
구분선 -3개
<>링크 혹은 [링크이름](링크주소)
