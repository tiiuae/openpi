import cv2

def main():
    # Open camera at index 16
    cap = cv2.VideoCapture(10)

    if not cap.isOpened():
        print("Error: Could not open camera at index 16")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Display the frame
        cv2.imshow("Camera Index 16", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
