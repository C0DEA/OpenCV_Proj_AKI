import cv2
import time

# [ADDED] Global variable to control whether faces are blurred
blur_enabled = False

# [ADDED] Mouse callback function to detect clicks on the blur button
def click_event(event, x, y, flags, param):
    global blur_enabled
    if event == cv2.EVENT_LBUTTONDOWN:
        # Coordinates of the blur button's rectangle
        # In this example, top-left: (10, 100), bottom-right: (110, 140)
        if 10 <= x <= 110 and 75 <= y <= 110:
            # Toggle the blur flag
            blur_enabled = not blur_enabled

def count_faces(faces):  # Returns the number of faces in the given list of face detections.
    return len(faces)

def main():
    # Path to the Haar Cascade file
    face_cascade_path = "haarcascade_frontalface_default.xml"
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Start video capture from your webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    
    # [ADDED] Named window + setting the mouse callback
    cv2.namedWindow('Face Detection')
    cv2.setMouseCallback('Face Detection', click_event)
    
    # -------------------------------
    # FPS-Related Variables:
    # -------------------------------
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Count the faces using our function
        face_count = count_faces(faces)
        
        # [CHANGED] Instead of always drawing rectangles, we check if blur is enabled
        for (x, y, w, h) in faces:
            if blur_enabled:
                # [ADDED] Blur the face region
                face_roi = frame[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 0)
                frame[y:y+h, x:x+w] = blurred_face
            else:
                # Original face rectangle drawing
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # ------------------------------------------------
        # Update FPS calculation
        # ------------------------------------------------
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # ------------------------------------------------
        # Display FPS on the frame
        # ------------------------------------------------
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        
        # ------------------------------------------------
        # Display face count on the frame
        # ------------------------------------------------
        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )
        
        # [ADDED] Draw the in-frame "blur button" rectangle and text
        button_color = (0, 200, 0) if blur_enabled else (0, 0, 200)
        cv2.rectangle(frame, (10, 70), (110, 95), button_color, -1)  # filled rectangle
        button_text = "Blur ON" if blur_enabled else "Blur OFF"
        cv2.putText(
            frame,
            button_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
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
