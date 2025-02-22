import cv2
import time  # <-- Import the time module to measure FPS

def main():
    # Path to the Haar Cascade file (must be correct!)
    face_cascade_path = "haarcascade_frontalface_default.xml"
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    
    # Start video capture from your webcam
    cap = cv2.VideoCapture(0)  # 0 = default camera
    
    cap.set(cv2.CAP_PROP_FPS, 60) #Ich dachte mir vllt ist die Webcam FPS rate von openCV restricted aber scheint nicht der fall zu sein.

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    # -------------------------------
    # FPS-Related Variables:
    # -------------------------------
    fps = 0              # Will store the current FPS
    frame_count = 0      # Counts the number of frames
    start_time = time.time()  # Start time for measuring FPS
    
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
        
        # ------------------------------------------------
        # Update FPS (frames per second) calculation
        # ------------------------------------------------
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # ------------------------------------------------
        # Display FPS on the frame
        # ------------------------------------------------
        cv2.putText(
            frame,                               # image
            f"FPS: {fps:.2f}",                   # text
            (10, 30),                            # position
            cv2.FONT_HERSHEY_SIMPLEX,            # font
            1.0,                                 # font scale
            (124, 255, 0),                         # color in BGR
            1,                                   # thickness
            cv2.LINE_AA                          # line type
        )
        
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
