### 🖐️ Hand-Control AI Gesture Recognizer

This project is a real-time hand gesture recognition application that uses your webcam to detect hand movements and control an interactive image on a web page. The backend, built with Python and Flask, uses MediaPipe for hand tracking, enabling live, dynamic interactions.

-----

### Features ✨

  * **Real-time Hand Tracking:** Uses your webcam to detect hand landmarks in real time.
  * **Object Interaction:** Pick up, move, and drop an image with specific hand gestures.
  * **Dynamic Size Control:** Adjust the image's size by moving two open hands apart or closer together.
  * **Image Duplication:** Duplicate an image using a specific two-handed gesture.
  * **Image Randomization:** Make a "Shaka" sign to randomize an image's position and size.
  * **Customizable Images:** The interactive object is an image, and you can easily change the image by updating the code and adding your own images to the `static` folder.

-----

### Installation and Setup 💻

To get the application up and running on your local machine, follow these steps:

#### Prerequisites

  * **Python:** Ensure you have Python 3.8 or a newer version installed.
  * **Webcam:** A functional webcam is required for hand detection.

#### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```sh
git clone https://github.com/PriyanshA0/Hand-Control.git
cd Hand-Control
```

#### Step 2: Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them using pip:

```sh
pip install -r requirements.txt
```

#### Step 3: Add Static Images

The application uses images for the interactive objects. Create a `static` folder in the project's root directory and add the image files you want to use. You can use your own images by naming them and placing them in the `static` folder, or by updating the `image_urls` list in `app.py`.

#### Step 4: Run the Application

Start the Flask application from the terminal:

```sh
python app.py
```

The application will launch on your local machine, and you will see a message indicating the server address, usually `http://127.0.0.1:5000`.

-----

### How to Use 🕹️

1.  **Open the Web Application:** Open your web browser and navigate to the server address provided in the terminal (e.g., `http://127.0.0.1:5000`).
2.  **Start the Camera:** Click the "**▶️ Start Camera**" button to activate your webcam. Your live camera feed will appear on the screen.
3.  **Interact with Gestures:** Use your hands in front of the camera to perform the gestures described below. The application will recognize your hand signs and control the interactive images accordingly.

#### Gesture List

| Gesture | Emoji | Description |
| :--- | :--- | :--- |
| **Open Palm** | 🖐️ | Pick up an object. |
| **Point** | ☝️ | Move a held object. |
| **Fist** | ✊ | Drop a held object. |
| **Thumbs Up**| 👍 | Make a held object bounce. |
| **Peace Sign**| ✌️ | Change the color of a held object. |
| **Shaka** | 🤙 | Randomize the position and size of all idle objects. |
| **Two-Hand Size**| 👐 | Use two open hands to increase/decrease an object's size by moving them apart/together. |
| **Two-Hand Duplicate**| 👯 | Use two Peace Signs to create a duplicate of an object. |
