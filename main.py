import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# =============All body landmarks Detected================
# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # all landmarks detected
#         results = holistic.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # all Face landmarks detected
#         mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
#
#         # all Right Hand landmarks detected
#         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         # all Left Hand landmarks detected
#         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         # all Pose landmarks detected
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#
#         cv2.imshow("Holistic Model Detection", image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# cap.release()
# cv2.destroyAllWindows()

# =============All body landmarks Detected================
# =============All body landmarks Detected and Style it================

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # all landmarks detected
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # all Face landmarks detected
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        #                           mp_drawing.DrawingSpec(color=(10, 10, 10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(10, 220, 10), thickness=1, circle_radius=1)
        #                           )

        # all Right Hand landmarks detected
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(230, 0, 0), thickness=2, circle_radius=2)
                                  )

        # all Left Hand landmarks detected
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(230, 0, 0), thickness=2, circle_radius=2)
                                  )

        # all Pose landmarks detected
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(10, 10, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(10, 220, 10), thickness=2, circle_radius=2)
                                  )

        cv2.imshow("Holistic Model Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
