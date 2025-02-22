import cv2
print(cv2.getBuildInformation())


def main():
    # Replace 'haarcascade_frontalface_default_cuda.xml' 
    # with the exact filename/path of your CUDA Haar model
    cascade_path = "haarcascade_frontalface_default_cuda.xml"
    
    # Create a CUDA-based CascadeClassifier
    face_cascade = cv2.cuda_CascadeClassifier(cascade_path)

    # Open default camera (change index if you have multiple cameras)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale (face detection typically expects grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upload grayscale image to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(gray)

        # Perform detection on GPU
        # Adjust scaleFactor and minNeighbors as needed for your use-case
        faces = face_cascade.detectMultiScale(
            gpu_frame,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # 'faces' is typically a tuple (rects, rejectLevels, levelWeights)
        # for the CUDA cascade in newer OpenCV versions, or just rects in older ones.
        # The first element in the tuple are the detected bounding boxes (rects).
        
        if isinstance(faces, tuple):
            # Newer CUDA cascade returns a tuple: (rects, rejectLevels, levelWeights)
            rects = faces[0]
        else:
            # Some versions return just the rectangles directly
            rects = faces
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the resulting frame
        cv2.imshow("CUDA Face Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
