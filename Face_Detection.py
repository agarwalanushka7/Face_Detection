import cv2

# Load the pre-trained face detection model
face_cap = cv2.CascadeClassifier(
    "C:/Users/agarw/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Open the webcam (0 = default camera)
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    
    if not ret:
        print("Failed to capture video")
        break

    # Convert the frame to grayscale (CORRECTED)
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("video_live", video_data)

    # To stop the video when 'a' is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release the camera and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
