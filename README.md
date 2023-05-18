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

<br />
<img width="924" alt="project pic" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/b2cfc158-d731-4279-b19a-7aaeb2e87c72">
<br />

1. Import Statements:
    - Importing required modules and classes from different Python files and libraries. 
   <br />

2. Yolov5 Thread Class:
    - A custom QThread subclass that runs YOLOv5 object detection on input frames.
    - The run method performs the main detection and annotation logic.
    - The send_img signal is emitted to send the annotated frames to the GUI.
    <br />

3. Deepsort Thread class:
    - Another custom QThread subclass that runs DeepSort object tracking on the detected objects from YOLOv5.
    - The run method performs the tracking and annotation logic.
    - The send_img2 signal is emitted to send the annotated frames to the GUI.
    <br />

4. Window Class:
    - Inherits from QWidget and represents the main application window.
    - Sets up the GUI elements, such as buttons, labels, and file dialogs.
    - Handles button clicks and connects them to appropriate actions.
    - Initializes instances of Yolov5Thread and DeepsortThread classes.
    - Contains methods for displaying images in the GUI labels.
    <br />

5. Main Execution:
    - Initializes the QApplication and creates an instance of the Window class.
    - Shows the window and starts the application event loop.
    <br />


## Analysis

📈 **This code creates a thread to run the YOLOv5 object detection model using Python. YOLOv5 is one of the deep learning models used for object detection in the field of computer vision.**

 - Yolov5 QThread
 <br />
 <br />
<img width="879" alt="yolo_init" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/90425de9-21d8-43ba-89bc-be00c47849b2">


> 🔎 **__init__**: In the initialization function, we set up the weights for the YOLOv5 's' model, the input source, the IOU threshold, and so on. This function is called only once when the thread is created.
<br />

<img width="880" alt="yolo_run" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/11381465-93d5-4ce3-9641-9da95b725c87">


> 🔎 **run**: This method is called when the thread starts, and the actual object detection operation is performed here. First, it loads the YOLOv5 model, processes the input image, and performs object detection using the model. Then, it uses the Annotator object to display information about the detected objects on the screen. Finally, it sends the processing results for each image via the send_img signal.
<br />

<img width="1109" alt="yolo_functions" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/6cca940b-0172-4567-87ae-79406d5ef398">


> 🔎 **attempt_load, check_img_size, non_max_suppression, scale_coords, Annotator, etc.**, are functions or classes provided by the YOLOv5 library. They are responsible for tasks such as model loading, image size checking, non-max suppression, coordinate adjustment, and adding annotations to the result.
> <br />

- DeepSort QThread
<br />
<br />
<img width="924" alt="deepsort_run" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/6907df2b-30b7-4282-a514-1cbb02117660">


> 🔎 **run**: This function is called when the thread starts, and the actual object detection and tracking work is performed here. It initializes DeepSORT and loads the YOLOv5 model. It then processes the input images and uses the model to detect objects, and then uses DeepSORT to track the detected objects. This operation is performed for all images in the input dataset, and tracking information is updated using the location information of the objects detected in the previous frame and the location information of the objects detected in the current frame. The tracked object's movement paths are stored in trajectories and this information is drawn as a line on the image.
<br />

<img width="777" alt="trajectory" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/31557050-0e38-48f3-8d96-46049e2daeca">


> 🔎 **Trajectory**: The code utilizes the object tracking results to draw the trajectory of each object. By connecting the previous and current positions with lines, it visualizes the movement path of the objects. The trajectory drawing is achieved using the cv2.line function from OpenCV.
<br />


<img width="885" alt="deepsort_func" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/879fb5f3-891d-4086-b32f-91b682948ef3">
<img width="794" alt="deepsort_func2" src="https://github.com/autJiahao/Real_Time_Track/assets/45887454/2f412db4-be50-4450-8853-fb744a707c09">


> 🔎 Functions or classes like **get_config, DeepSort, select_device, attempt_load, check_img_size, non_max_suppression, scale_coords, Annotator, xyxy2xywh, etc.**, are provided by the YOLOv5 or DeepSORT libraries. They are responsible for tasks such as model loading, image size checking, non-max suppression, coordinate adjustment, adding annotations to results, bbox conversion, etc.

----
헤더 #
줄바꿈 빈줄2줄
소스코드 위아래띄고 4줄인덴
구분선 -3개
<>링크 혹은 [링크이름](링크주소)
