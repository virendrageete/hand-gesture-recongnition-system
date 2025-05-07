import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(hand_landmarks, frame_height, frame_width):
    """
    Recognizes 15 gestures:
    - Open Hand
    - Fist
    - Victory (V symbol)
    - OK symbol
    - Thumbs Up
    - Pointing (Index)
    - Heart (Simplified, requires one hand)
    - Namaste
    - Thumb Up (Alternative)
    - ILY (I Love You)
    - Call Me
    - Vulcan Salute
    - Middle Finger
    - Fig Sign
    - Double Pointing (Index and Middle)
    """
    if hand_landmarks:
        landmarks = hand_landmarks.landmark

        # Get key landmark positions in pixel coordinates
        thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame_width, landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame_height])
        index_finger_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height])
        middle_finger_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_width, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_height])
        ring_finger_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame_width, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame_height])
        pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x * frame_width, landmarks[mp_hands.HandLandmark.PINKY_TIP].y * frame_height])

        thumb_base = np.array([landmarks[mp_hands.HandLandmark.THUMB_CMC].x * frame_width, landmarks[mp_hands.HandLandmark.THUMB_CMC].y * frame_height])
        index_finger_base = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * frame_width, landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame_height])
        middle_finger_base = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height])
        ring_finger_base = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x * frame_width, landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y * frame_height])
        pinky_base = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x * frame_width, landmarks[mp_hands.HandLandmark.PINKY_MCP].y * frame_height])

        # Helper function to check if fingers are curled
        def fingers_curled(tips, bases, threshold_factor=1.2):
            return all(np.linalg.norm(tip - base) < fist_threshold * threshold_factor for tip, base in zip(tips, bases))

        # Helper function to check if fingers are extended
        def fingers_extended(tips, bases, threshold_factor=0.8):
            return all(np.linalg.norm(tip - base) > open_hand_threshold * threshold_factor for tip, base in zip(tips, bases))

        # Helper function to check if two fingers are close
        def fingers_close(tip1, tip2, threshold_factor=1.5):
            return np.linalg.norm(tip1 - tip2) < ok_threshold * threshold_factor

        # Calculate distances
        thumb_index_dist = np.linalg.norm(thumb_tip - index_finger_tip)
        index_middle_dist = np.linalg.norm(index_finger_tip - middle_finger_tip)
        middle_ring_dist = np.linalg.norm(middle_finger_tip - ring_finger_tip)
        ring_pinky_dist = np.linalg.norm(ring_finger_tip - pinky_tip)

        open_hand_threshold = 0.1 * frame_height
        fist_threshold = 0.08 * frame_height
        victory_threshold = 0.1 * frame_height
        ok_threshold = 0.06 * frame_height
        thumbs_up_threshold = 0.1 * frame_height
        pointing_threshold = 0.1 * frame_height
        heart_threshold = 0.15 * frame_height
        namaste_threshold = 0.1 * frame_height
        thumb_up_alt_threshold = 0.1 * frame_height
        ily_threshold = 0.1 * frame_height
        call_threshold = 0.1 * frame_height
        vulcan_threshold = 0.1 * frame_height
        middle_finger_threshold = 0.15 * frame_height
        fig_threshold = 0.07 * frame_height
        double_point_threshold = 0.1 * frame_height

        # --- Gesture Detection Logic ---

        # 1. Open Hand: All fingers extended
        if (np.linalg.norm(index_finger_tip - index_finger_base) > open_hand_threshold and
                np.linalg.norm(middle_finger_tip - middle_finger_base) > open_hand_threshold and
                np.linalg.norm(ring_finger_tip - ring_finger_base) > open_hand_threshold and
                np.linalg.norm(pinky_tip - pinky_base) > open_hand_threshold and
                np.linalg.norm(thumb_tip - thumb_base) > open_hand_threshold):
            return "Open Hand"

        # 2. Fist: All fingers curled
        if (np.linalg.norm(index_finger_tip - index_finger_base) < fist_threshold and
                np.linalg.norm(middle_finger_tip - middle_finger_base) < fist_threshold and
                np.linalg.norm(ring_finger_tip - ring_finger_base) < fist_threshold and
                np.linalg.norm(pinky_tip - pinky_base) < fist_threshold and
                np.linalg.norm(thumb_tip - thumb_base) < fist_threshold * 1.5):
            return "Fist"

        # 3. Victory Sign: Index and middle fingers extended, others bent
        index_extended_vic = np.linalg.norm(index_finger_tip - index_finger_base) > victory_threshold
        middle_extended_vic = np.linalg.norm(middle_finger_tip - middle_finger_base) > victory_threshold
        ring_bent_vic = np.linalg.norm(ring_finger_tip - ring_finger_base) < victory_threshold
        pinky_bent_vic = np.linalg.norm(pinky_tip - pinky_base) < victory_threshold
        thumb_bent_vic = np.linalg.norm(thumb_tip - thumb_base) < victory_threshold
        if index_extended_vic and middle_extended_vic and ring_bent_vic and pinky_bent_vic and thumb_bent_vic:
            return "Victory"

        # 4. OK Sign: Thumb and index finger form a circle, other fingers extended
        ok_index_thumb_dist = np.linalg.norm(thumb_tip - index_finger_tip)
        ok_middle_extended = np.linalg.norm(middle_finger_tip - middle_finger_base) > ok_threshold
        ok_ring_extended = np.linalg.norm(ring_finger_tip - ring_finger_base) > ok_threshold
        ok_pinky_extended = np.linalg.norm(pinky_tip - pinky_base) > ok_threshold
        if ok_index_thumb_dist < ok_threshold and ok_middle_extended and ok_ring_extended and ok_pinky_extended:
            return "OK"

        # 5. Thumbs Up: Thumb extended upward, others bent
        thumb_extended_up = thumb_tip[1] < thumb_base[1] - thumbs_up_threshold
        index_bent_up = np.linalg.norm(index_finger_tip - index_finger_base) < thumbs_up_threshold
        middle_bent_up = np.linalg.norm(middle_finger_tip - middle_finger_base) < thumbs_up_threshold
        ring_bent_up = np.linalg.norm(ring_finger_tip - ring_finger_base) < thumbs_up_threshold
        pinky_bent_up = np.linalg.norm(pinky_tip - pinky_base) < thumbs_up_threshold
        if thumb_extended_up and index_bent_up and middle_bent_up and ring_bent_up and pinky_bent_up:
            return "Thumbs Up"

        # 6. Pointing (Index Finger): Index finger extended, others curled
        index_extended_point = np.linalg.norm(index_finger_tip - index_finger_base) > 1.5 * pointing_threshold
        thumb_bent_point = np.linalg.norm(thumb_tip - thumb_base) < pointing_threshold
        middle_bent_point = np.linalg.norm(middle_finger_tip - middle_finger_base) < pointing_threshold
        ring_bent_point = np.linalg.norm(ring_finger_tip - ring_finger_base) < pointing_threshold
        pinky_bent_point = np.linalg.norm(pinky_tip - pinky_base) < pointing_threshold
        if index_extended_point and thumb_bent_point and middle_bent_point and ring_bent_point and pinky_bent_point:
            return "Pointing (Index)"

        # 7. Heart (Simplified - Index and middle finger close)
        if index_middle_dist < heart_threshold and ring_pinky_dist > heart_threshold:
            return "Heart"

        # 8. Namaste: Palms together (requires two hands for accurate detection, this is a rough single-hand approximation)
        index_middle_dist_namaste = np.linalg.norm(index_finger_tip - middle_finger_tip)
        middle_ring_dist_namaste = np.linalg.norm(middle_finger_tip - ring_finger_tip)
        ring_pinky_dist_namaste = np.linalg.norm(ring_finger_tip - pinky_tip)
        thumb_index_dist_namaste = np.linalg.norm(thumb_tip - index_finger_tip)
        fingers_up_namaste = (index_finger_tip[1] < index_finger_base[1] and
                             middle_finger_tip[1] < middle_finger_base[1] and
                             ring_finger_tip[1] < ring_finger_base[1] and
                             pinky_tip[1] < pinky_base[1])
        thumb_close_to_index_namaste = thumb_index_dist_namaste < namaste_threshold * 1.5
        if fingers_up_namaste and thumb_close_to_index_namaste and index_middle_dist_namaste < namaste_threshold and middle_ring_dist_namaste < namaste_threshold and ring_pinky_dist_namaste < namaste_threshold:
            return "Namaste"

        # 9. Thumb Up (Alternative): Similar to the first one, might catch different angles
        thumb_pointing_up_alt = (thumb_tip[1] < thumb_base[1] - thumb_up_alt_threshold * 1.5)
        other_fingers_bent_alt = (np.linalg.norm(index_finger_tip - index_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(middle_finger_tip - middle_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(ring_finger_tip - ring_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(pinky_tip - pinky_base) < thumb_up_alt_threshold)
        if thumb_pointing_up_alt and other_fingers_bent_alt:
            return "Thumb Up"

        # 10. ILY (I Love You): Thumb, index, and pinky extended, middle and ring folded
        thumb_extended_ily = np.linalg.norm(thumb_tip - thumb_base) > ily_threshold
        index_extended_ily = np.linalg.norm(index_finger_tip - index_finger_base) > ily_threshold
        pinky_extended_ily = np.linalg.norm(pinky_tip - pinky_base) > ily_threshold
        middle_bent_ily = np.linalg.norm(middle_finger_tip - middle_finger_base) < ily_threshold
        ring_bent_ily = np.linalg.norm(ring_finger_tip - ring_finger_base) < ily_threshold
        if thumb_extended_ily and index_extended_ily and pinky_extended_ily and middle_bent_ily and ring_bent_ily:
            return "ILY"

        # 11. Call Me: Thumb and pinky extended, index and middle curled, ring might be curled
        thumb_extended_call = np.linalg.norm(thumb_tip - thumb_base) > call_threshold
        pinky_extended_call = np.linalg.norm(pinky_tip - pinky_base) > call_threshold
        index_bent_call = np.linalg.norm(index_finger_tip - index_finger_base) < call_threshold
        middle_bent_call = np.linalg.norm(middle_finger_tip - middle_finger_base) < call_threshold
        ring_curled_or_partially = np.linalg.norm(ring_finger_tip - ring_finger_base) < call_threshold * 1.2
        if thumb_extended_call and pinky_extended_call and index_bent_call and middle_bent_call and ring_curled_or_partially:
            return "Call Me"

        # 12. Vulcan Salute: Index and middle finger extended and separated, ring and pinky extended and together, thumb extended
        index_extended_vulcan = np.linalg.norm(index_finger_tip - index_finger_base) > vulcan_threshold
        middle_extended_vulcan = np.linalg.norm(middle_finger_tip - middle_finger_base) > vulcan_threshold
        ring_extended_vulcan = np.linalg.norm(ring_finger_tip - ring_finger_base) > vulcan_threshold
        pinky_extended_vulcan = np.linalg.norm(pinky_tip - pinky_base) > vulcan_threshold
        index_middle_far_vulcan = np.linalg.norm(index_finger_tip - middle_finger_tip) > vulcan_threshold * 0.8
        ring_pinky_close_vulcan = np.linalg.norm(ring_finger_tip - pinky_tip) < vulcan_threshold * 0.6
        thumb_extended_vulcan = np.linalg.norm(thumb_tip - thumb_base) > vulcan_threshold * 0.7
        if index_extended_vulcan and middle_extended_vulcan and ring_extended_vulcan and pinky_extended_vulcan and index_middle_far_vulcan and ring_pinky_close_vulcan and thumb_extended_vulcan:
            return "Vulcan Salute"

        # 13. Middle Finger: Only middle finger extended
        middle_extended_mf = np.linalg.norm(middle_finger_tip - middle_finger_base) > middle_finger_threshold
        index_bent_mf = np.linalg.norm(index_finger_tip - index_finger_base) < middle_finger_threshold
        ring_bent_mf = np.linalg.norm(ring_finger_tip - ring_finger_base) < middle_finger_threshold
        pinky_bent_mf = np.linalg.norm(pinky_tip - pinky_base) < middle_finger_threshold
        thumb_bent_mf = np.linalg.norm(thumb_tip - thumb_base) < middle_finger_threshold
        if middle_extended_mf and index_bent_mf and ring_bent_mf and pinky_bent_mf and thumb_bent_mf:
            return "Middle Finger"

        # 14. Fig Sign: Thumb tucked between index and middle finger (simplified)
        index_extended_fig = np.linalg.norm(index_finger_tip - index_finger_base) > fig_threshold
        middle_extended_fig = np.linalg.norm(middle_finger_tip - middle_finger_base) > fig_threshold
        thumb_between_index_middle = (thumb_tip[0] > index_finger_base[0] and thumb_tip[0] < middle_finger_base[0] and
                                     thumb_tip[1] > index_finger_base[1] and thumb_tip[1] < middle_finger_base[1])
        if index_extended_fig and middle_extended_fig and thumb_between_index_middle:
            return "Fig Sign"

        # 15. Double Pointing: Index and middle fingers extended
        index_extended_dp = np.linalg.norm(index_finger_tip - index_finger_base) > double_point_threshold
        middle_extended_dp = np.linalg.norm(middle_finger_tip - middle_finger_base) > double_point_threshold
        ring_bent_dp = np.linalg.norm(ring_finger_tip - ring_finger_base) < double_point_threshold
        pinky_bent_dp = np.linalg.norm(pinky_tip - pinky_base) < double_point_threshold
        thumb_not_extended_dp = np.linalg.norm(thumb_tip - thumb_base) < double_point_threshold * 1.5
        if index_extended_dp and middle_extended_dp and ring_bent_dp and pinky_bent_dp and thumb_not_extended_dp:
            return "Double Pointing"

    return "No Gesture"  # Default case if no gesture is recognized

def main():
    """
    Main function to capture video, process frames, and recognize gestures.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a more natural view
        frame_height, frame_width, _ = frame.shape  # Get frame dimensions

        # Convert the BGR image to RGB before processing with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)  # Process the frame with MediaPipe Hands

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks and connections

                gesture = recognize_gesture(hand_landmarks, frame_height, frame_width)  # Recognize the gesture
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display gesture

        cv2.imshow('Hand Gesture Recognition', frame)  # Show the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    main()  # Run the main function
