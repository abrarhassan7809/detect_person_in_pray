# ========shoulder and elbow angel detection=======
# import cv2
# import mediapipe as mp
# import numpy as np
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # ===counter_variables===
# counter = 0
# stage = None
#
#
# # ===function calculate angle===
# def calculate_left_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#
#     radians = np.arctan2(c[1] - b[-1], c[0] - b[0]) - np.arctan2(a[1] - b[-1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     angle = round(angle, 1)
#     print(angle)
#     if angle > 180.0:
#         angle = 360 - angle
#
#     return angle
#
#
# def calculate_right_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#
#     radians = np.arctan2(c[1] - b[-1], c[0] - b[0]) - np.arctan2(a[1] - b[-1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#
#     angle = round(angle, 1)
#     print(angle)
#     if angle > 180.0:
#         angle = 360 - angle
#
#     return angle
#
#
# # ====video feed===
# cap = cv2.VideoCapture(0)
# # ===media instance===
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         # ===recolor image===
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#
#         # ===make detection===
#         results = pose.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # ===extract landmark===
#         try:
#             landmarks = results.pose_landmarks.landmark
#
#             # ===get left coordinates===
#             left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#             left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
#
#             # ===get right coordinates===
#             right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#             right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#             right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
#                          landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

#
#             # ===calculate angle===
#             left_angle = calculate_left_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_right_angle(right_shoulder, right_elbow, right_wrist)
#
#             # ===visualize===
#             cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, str(right_angle), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#
#             # ===left angle counter logic===
#             if (left_angle and right_angle) > 160.0:
#                 stage = "Down"
#             if (left_angle and right_angle) < 30.0 and stage == "Down":
#                 stage = "Up"
#                 counter += 1
#
#         except Exception as e:
#             print(e)
#             pass
#
#         # ===setup texbox===
#         cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
#         cv2.putText(image, "REPS", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         # ===stage texbox===
#         cv2.putText(image, "STAGE", (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         # ===render detection===
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
#
#         cv2.imshow('Mediapipe Feed', image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

# ==========step one compleat==================

from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ===counter_variables===
counter = 0
stage = None


# ===function calculate angle===
def calculate_left_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[-1], c[0] - b[0]) - np.arctan2(a[1] - b[-1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    angle = round(angle, 1)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_right_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[-1], c[0] - b[0]) - np.arctan2(a[1] - b[-1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    angle = round(angle, 1)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# ====video feed===
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
# ===media instance===
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        hands, image = detector.findHands(frame)

        # ===recolor image===
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # ===make detection===
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ===extract landmark===
        try:
            landmarks = results.pose_landmarks.landmark

            # ===get left coordinates===
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # ===get right coordinates===
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # ===calculate angle===
            left_angle = calculate_left_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_right_angle(right_shoulder, right_elbow, right_wrist)

            # ===visualize===
            cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(right_angle), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # ===hand detect===
        if hands:
            hand1 = hands[0]
            if len(hands) == 2:
                hand2 = hands[1]
                hand1_fingers = detector.fingersUp(hand1)
                hand2_fingers = detector.fingersUp(hand2)

                # ===angle counter logic===
                # =========================
                if (left_angle and right_angle) > 150.0:
                    # print("Both Hands Down")
                    stage = "Down"

                if (hand1_fingers[0] and hand1_fingers[1] and hand1_fingers[2] and hand1_fingers[3] and hand1_fingers[
                    4]) and (
                        hand2_fingers[0] and hand2_fingers[1] and hand2_fingers[2] and hand2_fingers[3] and hand2_fingers[
                    4] and (left_angle and right_angle) < 40.0 and stage == "Down"):
                    # print("Both Hands Up")
                    stage = "Up"
                    counter += 1

        # ===steps counter===
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, "REPS", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ===stage detector===
        cv2.putText(image, "STAGE", (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ===pose landmarks===
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
