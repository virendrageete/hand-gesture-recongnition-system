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

def recognize_alphabet(hand_landmarks, frame_height, frame_width):
    """
    Recognizes alphabet hand gestures based on the provided image.
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

        # Define thresholds 
        a_threshold = 0.1 * frame_height
        b_threshold = 0.1 * frame_height
        c_threshold = 0.1 * frame_height
        d_threshold = 0.1 * frame_height
        e_threshold = 0.1 * frame_height
        f_threshold = 0.1 * frame_height
        g_threshold = 0.1 * frame_height
        h_threshold = 0.1 * frame_height
        i_threshold = 0.1 * frame_height
        k_threshold = 0.1 * frame_height
        l_threshold = 0.1 * frame_height
        m_threshold = 0.1 * frame_height
        n_threshold = 0.1 * frame_height
        o_threshold = 0.1 * frame_height
        p_threshold = 0.1 * frame_height
        q_threshold = 0.1 * frame_height
        r_threshold = 0.1 * frame_height
        s_threshold = 0.1 * frame_height
        t_threshold = 0.1 * frame_height
        u_threshold = 0.1 * frame_height
        v_threshold = 0.1 * frame_height
        w_threshold = 0.1 * frame_height
        x_threshold = 0.1 * frame_height
        y_threshold = 0.1 * frame_height

        # A: Fist with thumb extended to the side
        fist_like_a = (np.linalg.norm(index_finger_tip - index_finger_base) < a_threshold and
                     np.linalg.norm(middle_finger_tip - middle_finger_base) < a_threshold and
                     np.linalg.norm(ring_finger_tip - ring_finger_base) < a_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < a_threshold)
        thumb_extended_side_a = thumb_tip[0] > thumb_base[0] + 0.2 * frame_width  # Thumb position
        if fist_like_a and thumb_extended_side_a:
            return "A"

        # B: Fist with thumb across palm
        fist_like_b = (np.linalg.norm(index_finger_tip - index_finger_base) < b_threshold and
                     np.linalg.norm(middle_finger_tip - middle_finger_base) < b_threshold and
                     np.linalg.norm(ring_finger_tip - ring_finger_base) < b_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < b_threshold)
        thumb_across_palm_b = thumb_tip[0] < middle_finger_base[0]
        if fist_like_b and thumb_across_palm_b:
            return "B"

        # C: C shape with hand
        index_base_to_pinky_base_c = np.linalg.norm(index_finger_base - pinky_base)
        thumb_tip_to_pinky_tip_c = np.linalg.norm(thumb_tip - pinky_tip)
        if index_base_to_pinky_base_c > c_threshold and thumb_tip_to_pinky_tip_c > c_threshold:
            return "C"

        # D: Index finger pointing up, others bent
        index_extended_d = np.linalg.norm(index_finger_tip - index_finger_base) > d_threshold
        thumb_bent_d = np.linalg.norm(thumb_tip - thumb_base) < d_threshold
        middle_bent_d = np.linalg.norm(middle_finger_tip - middle_finger_base) < d_threshold
        ring_bent_d = np.linalg.norm(ring_finger_tip - ring_finger_base) < d_threshold
        pinky_bent_d = np.linalg.norm(pinky_tip - pinky_base) < d_threshold
        if index_extended_d and thumb_bent_d and middle_bent_d and ring_bent_d and pinky_bent_d:
            return "D"

        # E: All fingers curled into a fist shape
        fist_like_e = (np.linalg.norm(index_finger_tip - index_finger_base) < e_threshold and
                np.linalg.norm(middle_finger_tip - middle_finger_base) < e_threshold and
                np.linalg.norm(ring_finger_tip - ring_finger_base) < e_threshold and
                np.linalg.norm(pinky_tip - pinky_base) < e_threshold and
                np.linalg.norm(thumb_tip - thumb_base) < e_threshold)
        if fist_like_e:
            return "E"

        # F: Thumb and index finger make a circle, other fingers extended
        ok_index_thumb_dist_f = np.linalg.norm(thumb_tip - index_finger_tip)
        middle_extended_f = np.linalg.norm(middle_finger_tip - middle_finger_base) > f_threshold
        ring_extended_f = np.linalg.norm(ring_finger_tip - ring_finger_base) > f_threshold
        pinky_extended_f = np.linalg.norm(pinky_tip - pinky_base) > f_threshold
        if ok_index_thumb_dist_f < f_threshold and middle_extended_f and ring_extended_f and pinky_extended_f:
            return "F"

        # G: Index finger pointing out, thumb extended
        index_extended_g = np.linalg.norm(index_finger_tip - index_finger_base) > g_threshold
        thumb_extended_g = thumb_tip[0] > thumb_base[0] + 0.1 * frame_width
        middle_bent_g = np.linalg.norm(middle_finger_tip - middle_finger_base) < g_threshold
        ring_bent_g = np.linalg.norm(ring_finger_tip - ring_finger_base) < g_threshold
        pinky_bent_g = np.linalg.norm(pinky_tip - pinky_base) < g_threshold
        if index_extended_g and thumb_extended_g and middle_bent_g and ring_bent_g and pinky_bent_g:
            return "G"

        # H: Index and middle finger extended side by side, others bent
        index_extended_h = np.linalg.norm(index_finger_tip - index_finger_base) > h_threshold
        middle_extended_h = np.linalg.norm(middle_finger_tip - middle_finger_base) > h_threshold
        ring_bent_h = np.linalg.norm(ring_finger_tip - ring_finger_base) < h_threshold
        pinky_bent_h = np.linalg.norm(pinky_tip - pinky_base) < h_threshold
        thumb_bent_h = np.linalg.norm(thumb_tip - thumb_base) < h_threshold
        if index_extended_h and middle_extended_h and ring_bent_h and pinky_bent_h and thumb_bent_h:
            return "H"

        # I: Pinky finger extended, others bent
        pinky_extended_i = np.linalg.norm(pinky_tip - pinky_base) > i_threshold
        thumb_bent_i = np.linalg.norm(thumb_tip - thumb_base) < i_threshold
        index_bent_i = np.linalg.norm(index_finger_tip - index_finger_base) < i_threshold
        middle_bent_i = np.linalg.norm(middle_finger_tip - middle_finger_base) < i_threshold
        ring_bent_i = np.linalg.norm(ring_finger_tip - ring_finger_base) < i_threshold
        if pinky_extended_i and thumb_bent_i and index_bent_i and middle_bent_i and ring_bent_i:
            return "I"

        # K: Index and middle finger extended, thumb extended between
        index_extended_k = np.linalg.norm(index_finger_tip - index_finger_base) > k_threshold
        middle_extended_k = np.linalg.norm(middle_finger_tip - middle_finger_base) > k_threshold
        thumb_extended_k = thumb_tip[1] > thumb_base[1] - k_threshold
        ring_bent_k = np.linalg.norm(ring_finger_tip - ring_finger_base) < k_threshold
        pinky_bent_k = np.linalg.norm(pinky_tip - pinky_base) < k_threshold
        if index_extended_k and middle_extended_k and thumb_extended_k and ring_bent_k and pinky_bent_k:
            return "K"

        # L: Index finger extended up, thumb extended to the side
        index_extended_l = np.linalg.norm(index_finger_tip - index_finger_base) > l_threshold
        thumb_extended_side_l = thumb_tip[0] > thumb_base[0] + 0.1 * frame_width
        middle_bent_l = np.linalg.norm(middle_finger_tip - middle_finger_base) < l_threshold
        ring_bent_l = np.linalg.norm(ring_finger_tip - ring_finger_base) < l_threshold
        pinky_bent_l = np.linalg.norm(pinky_tip - pinky_base) < l_threshold
        if index_extended_l and thumb_extended_side_l and middle_bent_l and ring_bent_l and pinky_bent_l:
            return "L"

        # M: Thumb between pinky and ring finger, index and middle finger extended over thumb
        thumb_between_m = thumb_base[0] > pinky_base[0] and thumb_base[0] < middle_finger_base[0]
        index_extended_m = np.linalg.norm(index_finger_tip - index_finger_base) > m_threshold
        middle_extended_m = np.linalg.norm(middle_finger_tip - middle_finger_base) > m_threshold
        ring_bent_m = np.linalg.norm(ring_finger_tip - ring_finger_base) < m_threshold
        pinky_bent_m = np.linalg.norm(pinky_tip - pinky_base) < m_threshold
        if thumb_between_m and index_extended_m and middle_extended_m and ring_bent_m and pinky_bent_m:
            return "M"

        # N: Thumb between index and middle finger, ring and pinky finger extended over thumb
        thumb_between_n = thumb_base[0] > index_finger_base[0] and thumb_base[0] < middle_finger_base[0]
        ring_extended_n = np.linalg.norm(ring_finger_tip - ring_finger_base) > n_threshold
        pinky_extended_n = np.linalg.norm(pinky_tip - pinky_base) > n_threshold
        index_bent_n = np.linalg.norm(index_finger_tip - index_finger_base) < n_threshold
        middle_bent_n = np.linalg.norm(middle_finger_tip - middle_finger_base) < n_threshold
        if thumb_between_n and ring_extended_n and pinky_extended_n and index_bent_n and middle_bent_n:
            return "N"

        # O: Fingers curved
        index_curved_o = np.linalg.norm(index_finger_tip - index_finger_base) < o_threshold
        middle_curved_o = np.linalg.norm(middle_finger_tip - middle_finger_base) < o_threshold
        ring_curved_o = np.linalg.norm(ring_finger_tip - ring_finger_base) < o_threshold
        pinky_curved_o = np.linalg.norm(pinky_tip - pinky_base) < o_threshold
        thumb_curved_o = np.linalg.norm(thumb_tip - thumb_base) < o_threshold
        if index_curved_o and middle_curved_o and ring_curved_o and pinky_curved_o and thumb_curved_o:
            return "O"

        # P:  Fist, thumb and index finger extended down
        fist_like_p = (np.linalg.norm(middle_finger_tip - middle_finger_base) < p_threshold and
                     np.linalg.norm(ring_finger_tip - ring_finger_base) < p_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < p_threshold)
        thumb_down_p = thumb_tip[1] > thumb_base[1] + 0.1 * frame_height
        index_down_p = index_finger_tip[1] > index_finger_base[1] + 0.1 * frame_height
        if fist_like_p and thumb_down_p and index_down_p:
            return "P"

        # Q: Fist, thumb and index finger extended down
        fist_like_q = (np.linalg.norm(middle_finger_tip - middle_finger_base) < q_threshold and
                     np.linalg.norm(ring_finger_tip - ring_finger_base) < q_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < q_threshold)
        thumb_down_q = thumb_tip[1] > thumb_base[1] + 0.1 * frame_height
        index_down_q = index_finger_tip[1] > index_finger_base[1] + 0.1 * frame_height
        if fist_like_q and thumb_down_q and index_down_q:
            return "Q"

        # R: Index and middle finger crossed
        index_extended_r = np.linalg.norm(index_finger_tip - index_finger_base) > r_threshold
        middle_extended_r = np.linalg.norm(middle_finger_tip - middle_finger_base) > r_threshold
        index_middle_close_r = np.linalg.norm(index_finger_tip - middle_finger_tip) < r_threshold
        ring_bent_r = np.linalg.norm(ring_finger_tip - ring_finger_base) < r_threshold
        pinky_bent_r = np.linalg.norm(pinky_tip - pinky_base) < r_threshold
        thumb_bent_r = np.linalg.norm(thumb_tip - thumb_base) < r_threshold
        if index_extended_r and middle_extended_r and index_middle_close_r and ring_bent_r and pinky_bent_r and thumb_bent_r:
            return "R"

        # S: Fist with thumb over fingers
        fist_like_s = (np.linalg.norm(index_finger_tip - index_finger_base) < s_threshold and
                     np.linalg.norm(middle_finger_tip - middle_finger_base) < s_threshold and
                     np.linalg.norm(ring_finger_tip - ring_finger_base) < s_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < s_threshold)
        thumb_over_s = thumb_tip[1] < index_finger_base[1]
        if fist_like_s and thumb_over_s:
            return "S"

        # T: Fist with thumb between index and middle finger
        fist_like_t = (np.linalg.norm(ring_finger_tip - ring_finger_base) < t_threshold and
                     np.linalg.norm(pinky_tip - pinky_base) < t_threshold)
        thumb_between_t = thumb_base[0] > index_finger_base[0] and thumb_base[0] < middle_finger_base[0]
        index_extended_t = np.linalg.norm(index_finger_tip - index_finger_base) > t_threshold
        middle_extended_t = np.linalg.norm(middle_finger_tip - middle_finger_base) > t_threshold
        if fist_like_t and thumb_between_t and index_extended_t and middle_extended_t:
            return "T"

        # U: Index and middle finger extended and together
        index_extended_u = np.linalg.norm(index_finger_tip - index_finger_base) > u_threshold
        middle_extended_u = np.linalg.norm(middle_finger_tip - middle_finger_base) > u_threshold
        index_middle_close_u = np.linalg.norm(index_finger_tip - middle_finger_tip) < u_threshold
        ring_bent_u = np.linalg.norm(ring_finger_tip - ring_finger_base) < u_threshold
        pinky_bent_u = np.linalg.norm(pinky_tip - pinky_base) < u_threshold
        thumb_extended_u = np.linalg.norm(thumb_tip - thumb_base) > u_threshold
        if index_extended_u and middle_extended_u and index_middle_close_u and ring_bent_u and pinky_bent_u and thumb_extended_u:
            return "U"

        # V: Index and middle finger extended and apart
        index_extended_v = np.linalg.norm(index_finger_tip - index_finger_base) > v_threshold
        middle_extended_v = np.linalg.norm(middle_finger_tip - middle_finger_base) > v_threshold
        index_middle_far_v = np.linalg.norm(index_finger_tip - middle_finger_tip) > v_threshold
        ring_bent_v = np.linalg.norm(ring_finger_tip - ring_finger_base) < v_threshold
        pinky_bent_v = np.linalg.norm(pinky_tip - pinky_base) < v_threshold
        thumb_bent_v = np.linalg.norm(thumb_tip - thumb_base) < v_threshold
        if index_extended_v and middle_extended_v and index_middle_far_v and ring_bent_v and pinky_bent_v and thumb_bent_v:
            return "V"

        # W: Index, middle and ring finger extended
        index_extended_w = np.linalg.norm(index_finger_tip - index_finger_base) > w_threshold
        middle_extended_w = np.linalg.norm(middle_finger_tip - middle_finger_base) > w_threshold
        ring_extended_w = np.linalg.norm(ring_finger_tip - ring_finger_base) > w_threshold
        pinky_bent_w = np.linalg.norm(pinky_tip - pinky_base) < w_threshold
        thumb_extended_w = np.linalg.norm(thumb_tip - thumb_base) > w_threshold
        if index_extended_w and middle_extended_w and ring_extended_w and pinky_bent_w and thumb_extended_w:
            return "W"

        # X: Index finger curved
        index_curved_x = np.linalg.norm(index_finger_tip - index_finger_base) < x_threshold
        middle_bent_x = np.linalg.norm(middle_finger_tip - middle_finger_base) < x_threshold
        ring_bent_x = np.linalg.norm(ring_finger_tip - ring_finger_base) < x_threshold
        pinky_bent_x = np.linalg.norm(pinky_tip - pinky_base) < x_threshold
        thumb_extended_x = np.linalg.norm(thumb_tip - thumb_base) > x_threshold
        if index_curved_x and middle_bent_x and ring_bent_x and pinky_bent_x and thumb_extended_x:
            return "X"

        # Y: Thumb and pinky extended
        thumb_extended_y = np.linalg.norm(thumb_tip - thumb_base) > y_threshold
        pinky_extended_y = np.linalg.norm(pinky_tip - pinky_base) > y_threshold
        index_bent_y = np.linalg.norm(index_finger_tip - index_finger_base) < y_threshold
        middle_bent_y = np.linalg.norm(middle_finger_tip - middle_finger_base) < y_threshold
        ring_bent_y = np.linalg.norm(ring_finger_tip - ring_finger_base) < y_threshold
        if thumb_extended_y and pinky_extended_y and index_bent_y and middle_bent_y and ring_bent_y:
            return "Y"

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
                gesture = recognize_alphabet(hand_landmarks, frame_height, frame_width)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break theloop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
