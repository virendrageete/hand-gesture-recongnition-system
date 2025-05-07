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
    Recognizes 12 gestures:
    - Open Hand
    - Fist
    - Victory (V symbol)
    - OK symbol
    - Thumbs Up
    - Pointing (Index)
    - Rock
    - Heart (Simplified, requires one hand)
    - Namaste
    - Thumb Up (Alternative)
    - ILY (I Love You)
    - Call Me
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

        # Calculate distances (you might need to experiment with thresholds)
        thumb_index_dist = np.linalg.norm(thumb_tip - index_finger_tip)
        index_middle_dist = np.linalg.norm(index_finger_tip - middle_finger_tip)
        middle_ring_dist = np.linalg.norm(middle_finger_tip - ring_finger_tip)
        ring_pinky_dist = np.linalg.norm(ring_finger_tip - pinky_tip)

        # --- New Gesture Detection Logic ---

        # 1. Open Hand: All fingers extended
        open_hand_threshold = 0.1 * frame_height
        if (np.linalg.norm(index_finger_tip - index_finger_base) > open_hand_threshold and
                np.linalg.norm(middle_finger_tip - middle_finger_base) > open_hand_threshold and
                np.linalg.norm(ring_finger_tip - ring_finger_base) > open_hand_threshold and
                np.linalg.norm(pinky_tip - pinky_base) > open_hand_threshold and
                np.linalg.norm(thumb_tip - thumb_base) > open_hand_threshold): # Added thumb check
            return "Open Hand"

        # 2. Fist: All fingers curled
        fist_threshold = 0.08 * frame_height
        if (np.linalg.norm(index_finger_tip - index_finger_base) < fist_threshold and
                np.linalg.norm(middle_finger_tip - middle_finger_base) < fist_threshold and
                np.linalg.norm(ring_finger_tip - ring_finger_base) < fist_threshold and
                np.linalg.norm(pinky_tip - pinky_base) < fist_threshold and
                np.linalg.norm(thumb_tip - thumb_base) < fist_threshold * 1.5): # Relaxed thumb a bit
            return "Fist"

        # 3. Victory Sign: Index and middle fingers extended, others bent
        victory_threshold = 0.1 * frame_height
        index_extended = np.linalg.norm(index_finger_tip - index_finger_base) > victory_threshold
        middle_extended = np.linalg.norm(middle_finger_tip - middle_finger_base) > victory_threshold
        ring_bent = np.linalg.norm(ring_finger_tip - ring_finger_base) < victory_threshold
        pinky_bent = np.linalg.norm(pinky_tip - pinky_base) < victory_threshold
        thumb_bent_vic = np.linalg.norm(thumb_tip - thumb_base) < victory_threshold # Added thumb check
        if index_extended and middle_extended and ring_bent and pinky_bent and thumb_bent_vic:
            return "Victory"

        # 4. OK Sign: Thumb and index finger form a circle, other fingers extended
        ok_threshold = 0.06 * frame_height
        ok_index_thumb_dist = np.linalg.norm(thumb_tip - index_finger_tip)
        ok_middle_extended = np.linalg.norm(middle_finger_tip - middle_finger_base) > ok_threshold
        ok_ring_extended = np.linalg.norm(ring_finger_tip - ring_finger_base) > ok_threshold
        ok_pinky_extended = np.linalg.norm(pinky_tip - pinky_base) > ok_threshold
        if ok_index_thumb_dist < ok_threshold and ok_middle_extended and ok_ring_extended and ok_pinky_extended:
            return "OK"

        # 5. Thumbs Up: Thumb extended upward, others bent
        thumbs_up_threshold = 0.1 * frame_height
        thumb_extended_up = thumb_tip[1] < thumb_base[1] - thumbs_up_threshold
        index_bent_up = np.linalg.norm(index_finger_tip - index_finger_base) < thumbs_up_threshold
        middle_bent_up = np.linalg.norm(middle_finger_tip - middle_finger_base) < thumbs_up_threshold
        ring_bent_up = np.linalg.norm(ring_finger_tip - ring_finger_base) < thumbs_up_threshold
        pinky_bent_up = np.linalg.norm(pinky_tip - pinky_base) < thumbs_up_threshold
        if thumb_extended_up and index_bent_up and middle_bent_up and ring_bent_up and pinky_bent_up:
            return "Thumbs Up"

        # 6. Pointing (Index Finger): Index finger extended, others curled
        pointing_threshold = 0.1 * frame_height
        index_extended_point = np.linalg.norm(index_finger_tip - index_finger_base) > 1.5 * pointing_threshold # More strict
        thumb_bent_point = np.linalg.norm(thumb_tip - thumb_base) < pointing_threshold
        middle_bent_point = np.linalg.norm(middle_finger_tip - middle_finger_base) < pointing_threshold
        ring_bent_point = np.linalg.norm(ring_finger_tip - ring_finger_base) < pointing_threshold
        pinky_bent_point = np.linalg.norm(pinky_tip - pinky_base) < pointing_threshold
        if index_extended_point and thumb_bent_point and middle_bent_point and ring_bent_point and pinky_bent_point:
            return "Pointing (Index)"

        # 7. Rock: Fist with thumb extended upward or to the side
        rock_threshold = 0.1 * frame_height
        fist_like_rock = (np.linalg.norm(index_finger_tip - index_finger_base) < rock_threshold and
                         np.linalg.norm(middle_finger_tip - middle_finger_base) < rock_threshold and
                         np.linalg.norm(ring_finger_tip - ring_finger_base) < rock_threshold and
                         np.linalg.norm(pinky_tip - pinky_base) < rock_threshold)
        thumb_extended_rock = np.linalg.norm(thumb_tip - thumb_base) > rock_threshold
        if fist_like_rock and thumb_extended_rock:
            return "Rock"

        # 8. Heart (Simplified - Index and middle finger close)
        heart_threshold = 0.15 * frame_height
        if index_middle_dist < heart_threshold and ring_pinky_dist > heart_threshold: # Added condition for other fingers
            return "Heart"

        # 9. Namaste: Palms together (requires two hands for accurate detection, this is a rough single-hand approximation)
        namaste_threshold = 0.1 * frame_height
        index_middle_dist_namaste = np.linalg.norm(index_finger_tip - middle_finger_tip)
        middle_ring_dist_namaste = np.linalg.norm(middle_finger_tip - ring_finger_tip)
        ring_pinky_dist_namaste = np.linalg.norm(ring_finger_tip - pinky_tip)
        thumb_index_dist_namaste = np.linalg.norm(thumb_tip - index_finger_tip)
        fingers_up = (index_finger_tip[1] < index_finger_base[1] and
                      middle_finger_tip[1] < middle_finger_base[1] and
                      ring_finger_tip[1] < ring_finger_base[1] and
                      pinky_tip[1] < pinky_base[1])
        thumb_close_to_index = thumb_index_dist_namaste < namaste_threshold * 1.5
        if fingers_up and thumb_close_to_index and index_middle_dist_namaste < namaste_threshold and middle_ring_dist_namaste < namaste_threshold and ring_pinky_dist_namaste < namaste_threshold:
            return "Namaste"

        # 10. Thumb Up (Alternative): Similar to the first one, might catch different angles
        thumb_up_alt_threshold = 0.1 * frame_height
        thumb_pointing_up_alt = (thumb_tip[1] < thumb_base[1] - thumb_up_alt_threshold * 1.5)
        other_fingers_bent_alt = (np.linalg.norm(index_finger_tip - index_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(middle_finger_tip - middle_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(ring_finger_tip - ring_finger_base) < thumb_up_alt_threshold and
                                 np.linalg.norm(pinky_tip - pinky_base) < thumb_up_alt_threshold)
        if thumb_pointing_up_alt and other_fingers_bent_alt:
            return "Thumb Up"

        # --- New Gestures ---

        # 11. ILY (I Love You): Thumb, index, and pinky extended, middle and ring folded
        ily_threshold = 0.1 * frame_height
        thumb_extended_ily = np.linalg.norm(thumb_tip - thumb_base) > ily_threshold
        index_extended_ily = np.linalg.norm(index_finger_tip - index_finger_base) > ily_threshold
        pinky_extended_ily = np.linalg.norm(pinky_tip - pinky_base) > ily_threshold
        middle_bent_ily = np.linalg.norm(middle_finger_tip - middle_finger_base) < ily_threshold
        ring_bent_ily = np.linalg.norm(ring_finger_tip - ring_finger_base) < ily_threshold
        if thumb_extended_ily and index_extended_ily and pinky_extended_ily and middle_bent_ily and ring_bent_ily:
            return "ILY"

        # 12. Call Me: Thumb and pinky extended, index and middle curled, ring might be curled
        call_threshold = 0.1 * frame_height
        thumb_extended_call = np.linalg.norm(thumb_tip - thumb_base) > call_threshold
        pinky_extended_call = np.linalg.norm(pinky_tip - pinky_base) > call_threshold
        index_bent_call = np.linalg.norm(index_finger_tip - index_finger_base) < call_threshold
        middle_bent_call = np.linalg.norm(middle_finger_tip - middle_finger_base) < call_threshold
        # Ring finger can be in a more ambiguous state, so we might relax the condition
        ring_curled_or_partially = np.linalg.norm(ring_finger_tip - ring_finger_base) < call_threshold * 1.2
        if thumb_extended_call and pinky_extended_call and index_bent_call and middle_bent_call and ring_curled_or_partially:
            return "Call Me"

        # Default if no specific gesture is recognized
        return "No Specific Gesture"
    return "No Hand Detected"

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # If hands are detected, draw landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks, frame_height, frame_width)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()