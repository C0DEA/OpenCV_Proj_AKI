import cv2

def main():
    # Path to the Haar Cascade file (must be correct!)
    face_cascade_path = "haarcascade_frontalface_default.xml"
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Start video capture from your webcam
    cap = cv2.VideoCapture(0)  # 0 = default camera
    
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (adjust scaleFactor, minNeighbors if needed)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Face Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()