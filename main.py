import cv2
import time
import mediapipe as mp
from utils.math_stuff import get_all_finger_angles
from utils.hand_stuff import get_hand_keypoint, hand_keypoint_to_pixel_c, draw_circle, draw_line

MODEL_ASSET_PATH = "hand_landmarker.task"
NUM_HANDS = 2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=NUM_HANDS
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape

        # Convert the frame to MediaPipe's Image format.
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect landmarks (timestamp is required for VIDEO mode).
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                hand_keypoints = get_hand_keypoint(hand_landmarks)

                # Convert to Pixel Coordinates.
                hand_keypoint_pixels = hand_keypoint_to_pixel_c(hand_keypoints, w, h)
                
                afa = get_all_finger_angles(hand_keypoint_pixels)
                for angle, (t, other_t, midpoint) in afa:
                    draw_circle(frame, t)
                    draw_circle(frame, other_t)
                    draw_circle(frame, midpoint)

                    draw_line(frame, t, midpoint)
                    draw_line(frame, other_t, midpoint)

                    # Put the angles on screen.
                    cv2.putText(
                        frame, f"{int(angle)}", (midpoint[0], midpoint[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                    )

        cv2.imshow("CAM", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
