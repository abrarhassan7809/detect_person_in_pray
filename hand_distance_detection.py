# import os
# import time
# import cvzone
# import mediapipe as mp
# import cv2
#
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
#
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
#
# =============All body landmarks Detected================
# ========All body landmarks Detected and Style it========
#
# pTime = 0
#
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
#         # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#         #                           mp_drawing.DrawingSpec(color=(10, 10, 10), thickness=1, circle_radius=1),
#         #                           mp_drawing.DrawingSpec(color=(10, 220, 10), thickness=1, circle_radius=1)
#         #                           )
#
#         # # all Right Hand landmarks detected
#         # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#         #                           mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
#         #                           mp_drawing.DrawingSpec(color=(230, 0, 0), thickness=2, circle_radius=2)
#         #                           )
#         #
#         # # all Left Hand landmarks detected
#         # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#         #                           mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
#         #                           mp_drawing.DrawingSpec(color=(230, 0, 0), thickness=2, circle_radius=2)
#         #                           )
#         #
#         # # all Pose landmarks detected
#         # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#         #                           mp_drawing.DrawingSpec(color=(10, 10, 10), thickness=2, circle_radius=2),
#         #                           mp_drawing.DrawingSpec(color=(10, 220, 10), thickness=2, circle_radius=2)
#         #                           )
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

# =============hand angel detector==================
#
# import cv2
# from cvzone.HandTrackingModule import HandDetector
#
# pTime = 0
# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# while cap.isOpened():
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#
#     if hands:
#         hand1 = hands[0]
#         lmList1 = hand1["lmList"]
#         bbox1 = hand1["bbox"]
#         centerPoint1 = hand1["center"]
#         handType1 = hand1["type"]
#
#         hand1_fingers = detector.fingersUp(hand1)
#
#         if len(hands) == 2:
#             hand2 = hands[1]
#             lmList2 = hand2["lmList"]
#             bbox2 = hand2["bbox"]
#             centerPoint2 = hand2["center"]
#             handType2 = hand2["type"]
#
#             hand2_fingers = detector.fingersUp(hand2)
#
#             if (hand1_fingers[0] & hand1_fingers[1] & hand1_fingers[2] & hand1_fingers[3] & hand1_fingers[4]) & (hand2_fingers[0] & hand2_fingers[1] & hand2_fingers[2] & hand2_fingers[3] & hand2_fingers[4]):
#                 print("Both Hands Fingers Up")
#
#             elif not (hand1_fingers[0] & hand1_fingers[1] & hand1_fingers[2] & hand1_fingers[3] & hand1_fingers[4]) & (hand2_fingers[0] & hand2_fingers[1] & hand2_fingers[2] & hand2_fingers[3] & hand2_fingers[4]):
#                 print("No Hands Fingers Up")
#
#             # 1st finger distance
#             # length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)
#
#             # center of hand distance
#             length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)
#
#             cv2.putText(img, f'Distance: {round(length, 1)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#
#     cv2.imshow("Hand Detection", img)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# =============hand distance detector==================
import cv2
import mediapipe as mp
import numpy as np
import math

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


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


def calculate_right_hd_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    distance = math.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
    distance1 = (distance * 10) + distance - (distance * 10)
    distance = distance1

    return distance


def calculate_left_hd_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    distance = math.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
    distance1 = (distance * 10) + distance - (distance * 10)
    distance = distance1

    return distance


# ====video feed===
stage = None
counter = 0
cap = cv2.VideoCapture(0)
# ===media instance===
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1200, 720))

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
            left_hand_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            left_foot_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

            # ===get right coordinates===
            right_hand_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            right_foot_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

            # ===calculate angle===
            left_elbow_angle = calculate_left_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_right_angle(right_shoulder, right_elbow, right_wrist)
            # ---------
            left_knee_angle = calculate_left_angle(left_hip, left_knee, left_foot)
            right_knee_angle = calculate_left_angle(right_hip, right_knee, right_foot)
            # ---------
            left_hip_angle = calculate_left_angle(left_shoulder, left_hip, left_foot)
            right_hip_angle = calculate_left_angle(right_shoulder, right_hip, right_foot)
            # ---------
            left_hip_kn_angle = calculate_left_angle(left_shoulder, left_hip, left_knee)
            right_hip_kn_angle = calculate_left_angle(right_shoulder, right_hip, right_knee)
            # ---------
            # print(right_foot_distance)

            # ===calculate distance===
            left_hd_ear_distance = calculate_left_hd_distance(left_hand_thumb, left_ear)
            right_hd_ear_distance = calculate_right_hd_distance(right_hand_thumb, right_ear)
            # ---------
            lh_rh_wrists_distance = calculate_right_hd_distance(right_wrist, left_wrist)
            # ---------
            left_kn_hd_distance = calculate_left_hd_distance(left_knee, left_hand_thumb)
            right_kn_hd_distance = calculate_right_hd_distance(right_knee, right_hand_thumb)
            # ---------
            left_hip_ft_distance = calculate_left_hd_distance(left_foot, left_hip)
            right_hip_ft_distance = calculate_right_hd_distance(right_foot, right_hip)

            # ===visualize text===
            # cv2.putText(image, str(left_knee_angle), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str(right_knee_angle), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, f'Right: {round(right_elbow_angle, 1)}', (30, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
            #             2)
            # cv2.putText(image, f'Left: {round(left_elbow_angle, 1)}', (450, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
            #             2)
            # cv2.putText(image, f'Left: {counter}', (450, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
            #             2)
            cv2.putText(image, f'Left: {left_elbow_angle}', (450, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                        2)
            # cv2.putText(image, f'Right: {right_elbow_angle}', (30, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
            #             2)

        except Exception as e:
            print(e)
            pass

        try:
            # ==============step one==========
            if left_elbow_angle < 25 and right_elbow_angle < 25 and left_hd_ear_distance < 1.5 and right_hd_ear_distance < 1.5 and stage is None:
                stage = "Taqbir"
                counter = 1

            # ==============step two==========
            if left_elbow_angle > 120 and right_elbow_angle > 120 and lh_rh_wrists_distance < 1.0 and stage == "Taqbir" and counter == 1:
                stage = "Qayam"
                counter = 2

            # ==============step three==========
            if left_hip_angle < 120 and right_hip_angle < 120 and right_kn_hd_distance < 1.0 and left_kn_hd_distance < 1.0 and counter == 2 and stage == "Qayam":
                stage = "Raku"
                counter = 3

            # ==============step four===========
            if left_hip_angle > 150 and right_hip_angle > 150 and counter == 3 and stage == "Raku":
                stage = "Qayam1"
                counter = 4

            # ==============step five==========
            if left_knee_angle < 80.0 and right_knee_angle < 80.0 and counter == 4 and stage == "Qayam1":
                stage = "Sajud"
                counter = 5

            if left_knee_angle > 90.0 and right_knee_angle > 90.0 and counter == 5 and stage == "Sajud":
                stage = "Person in Prayer"
                counter = 6

        except:
            pass

        # ==============step four==========

        cv2.putText(image, "STAGE:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, stage, (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
