
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  
</head>
<body>

  <h1>✋ Sign Language Detection using CNN</h1>
  <p>
   This project is a comprehensive system for sign language detection that combines real-time data collection, preprocessing, model training, and gesture recognition using computer vision and deep learning techniques. The aim is to provide an efficient solution for recognizing hand gestures corresponding to the English alphabet, enabling seamless communication for individuals relying on sign language.

The data collection phase utilizes a webcam to capture live hand gesture images. The region of interest (ROI) for the hand is isolated, followed by preprocessing techniques such as grayscale conversion, Gaussian blurring, and adaptive thresholding to enhance the quality of the images. The processed images are organized into 26 gesture classes corresponding to the English alphabet, forming the training and testing datasets.

For the detection task, a Convolutional Neural Network (CNN) is implemented using Keras. The architecture features multiple convolutional and pooling layers for feature extraction, followed by flattening and fully connected dense layers for classification. The final softmax layer predicts one of the 26 gesture categories. To improve model robustness and accuracy, data augmentation techniques, including rescaling, rotation, and zoom, are applied to the training data. The model is optimized using the Adam optimizer with a categorical cross-entropy loss function and achieves significant performance gains across 50 training epochs. Evaluation metrics such as training and validation accuracy and loss trends are visualized to monitor model performance.

In the detection phase, the trained CNN model processes live webcam feeds to detect and classify hand gestures in real time. The system extracts the hand ROI from each frame, applies the trained model to predict the corresponding sign language gesture, and overlays the predicted alphabet on the video feed. This provides a user-friendly interface for real-time gesture recognition. 

This project offers a robust and efficient solution for detecting and interpreting sign language gestures, paving the way for practical applications in communication and accessibility technologies.
  </p>

  <h2>📌 Key Features</h2>
  <ul>
    <li>Real-time hand gesture detection via webcam</li>
    <li>26-class CNN model trained on A-Z gestures</li>
    <li>Data augmentation for improved generalization</li>
    <li>Live video classification with gesture overlay</li>
    <li>Performance visualization through accuracy/loss plots</li>
  </ul>

  <h2>📂 Data Collection & Preprocessing</h2>
  <ul>
    <li>Hand ROI captured from webcam</li>
    <li>Preprocessing steps:
      <ul>
        <li>Grayscale conversion</li>
        <li>Gaussian blurring</li>
        <li>Adaptive thresholding</li>
      </ul>
    </li>
    <li>Organized into folders labeled A-Z for model training</li>
  </ul>

  <h2>🧠 Model Architecture (CNN)</h2>
  <ul>
    <li>Three convolutional layers with ReLU activations</li>
    <li>Max pooling after each convolutional block</li>
    <li>Dense layers with Dropout for regularization</li>
    <li>Softmax output layer for 26 gesture classes</li>
    <li>Trained for 50 epochs using Adam optimizer</li>
  </ul>

  <h3>🔧 Model Code (Keras)</h3>
  <pre><code>classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  </code></pre>

  <h2>🧪 Training & Evaluation</h2>
  <ul>
    <li>Train-test split with ImageDataGenerator</li>
    <li>Augmentations: rotation, zoom, shift, rescale</li>
    <li>Training accuracy > 95% after 50 epochs</li>
    <li>Real-time classification validated with webcam demo</li>
  </ul>

  <h2>📊 Visualizations</h2>
  <p>Training and validation accuracy over epochs:</p>
  <img src="assets/train_accuracy.png" alt="Training Accuracy Plot">

  <p>Training and validation loss over epochs:</p>
  <img src="assets/train_loss.png" alt="Training Loss Plot">

  <h2>🎯 Real-time Gesture Recognition</h2>
  <ul>
    <li>Live webcam feed processed frame by frame</li>
    <li>ROI extracted and passed to trained model</li>
    <li>Predicted letter is overlaid on video</li>
    <li>Responsive and intuitive user experience</li>
  </ul>

  <h2>🚀 Future Work</h2>
  <ul>
    <li>Add support for dynamic gestures and words</li>
    <li>Deploy as a web app using TensorFlow.js or Streamlit</li>
    <li>Integrate speech synthesis for predicted letters</li>
    <li>Expand to ASL or other regional sign languages</li>
  </ul>

  <h2>📬 Contact</h2>
  <p>
    For contributions, suggestions, or collaboration, feel free to connect via
    <a href="https://github.com/your-username">GitHub</a> or email.
  </p>

</body>
</html>
