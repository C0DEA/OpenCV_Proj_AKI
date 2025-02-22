import cv2
import time

# Global variable to track whether blurring is enabled
blur_enabled = False

def click_event(event, x, y, flags, param):
    """
    Mouse callback to detect clicks on our 'blur' button.
    """
    global blur_enabled
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Coordinates of our button (x1, y1, x2, y2)
        # In this example, let's define the button's top-left corner at (10, 50)
        # and its bottom-right corner at (110, 90).
        if 10 <= x <= 110 and 50 <= y <= 90:
            # Toggle the blur on/off
            blur_enabled = not blur_enabled

def main():
    # Path to Haar Cascade
    face_cascade_path = "haarcascade_frontalface_default.xml"
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Start video capture (webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    
    # Set a named window and assign the mouse callback to it
    cv2.namedWindow('Face Detection')
    cv2.setMouseCallback('Face Detection', click_event)
    
    # FPS variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If blur is enabled, blur faces; otherwise draw green rectangles
        for (x, y, w, h) in faces:
            if blur_enabled:
                # Extract face region of interest
                face_roi = frame[y:y+h, x:x+w]
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 0)
                frame[y:y+h, x:x+w] = blurred_face
            else:
                # Just draw rectangles
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS
        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # ================================
        # Draw Button on Frame
        # ================================
        # Button rectangle
        cv2.rectangle(frame, (10, 50), (110, 90), (200, 0, 0), -1)  # filled rectangle (blue-ish)
        
        # Button text
        # Show "Blur ON" or "Blur OFF" to indicate current state
        button_text = "Blur ON" if blur_enabled else "Blur OFF"
        cv2.putText(
            frame, button_text, (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Show the frame
        cv2.imshow('Face Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
