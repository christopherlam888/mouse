#!/usr/bin/env python3

import mediapipe as mp
import cv2
import pyautogui
import os

PARENT_PATH = os.path.dirname(__file__)
MODEL_PATH = "gesture_recognizer.task"
FULL_PATH = os.path.join(PARENT_PATH, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

cursor = False


# Identify the tip of the index finger
def get_index_landmark(result):
    INDEX_LANDMARK = 8
    if result.hand_landmarks:
        hand_landmarks_hands = result.hand_landmarks
        hand_landmarks_list = hand_landmarks_hands[0]
        return hand_landmarks_list[INDEX_LANDMARK]


def get_gesture(result):
    if result.gestures:
        return result.gestures[0][0].category_name


# Print the result of the hand gesture
def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    global cursor

    if cursor:
        if get_index_landmark(result):
            # calculate horizontal and vertical movement
            delta_x = get_index_landmark(result).x - 0.5
            delta_y = get_index_landmark(result).y - 0.5

            # horizontal movement greater than vertical movement
            if abs(delta_x) > abs(delta_y):
                if delta_x > 0:
                    print("right")
                    pyautogui.moveRel(10, 0)
                else:
                    print("left")
                    pyautogui.moveRel(-10, 0)
            # vertical movement greater than horizontal movement
            else:
                if delta_y > 0:
                    print("down")
                    pyautogui.moveRel(0, 10)
                else:
                    print("up")
                    pyautogui.moveRel(0, -10)

    if get_gesture(result) == "Closed_Fist":
        cursor ^= True
        print(cursor)

    if get_gesture(result) == "Pointing_Up":
        print("click")
        pyautogui.click()

    if get_gesture(result) == "Thumb_Up":
        print("scroll up")
        pyautogui.scroll(1)

    if get_gesture(result) == "Thumb_Down":
        print("scroll down")
        pyautogui.scroll(-1)


# Start the hand landmarker
def run():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=FULL_PATH),
        num_hands=1,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )

    with GestureRecognizer.create_from_options(options) as landmarker:
        vid = cv2.VideoCapture(0)

        while True:
            # read the video frame
            ret, frame = vid.read()

            # mirror
            frame = cv2.flip(frame, 1)

            # format frame to mp image and detect hand landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.recognize_async(mp_image, int(vid.get(cv2.CAP_PROP_POS_MSEC)))

            # draw green lines separating directions for live video view
            # height, width, _ = frame.shape
            # cv2.line(frame, (0, 0), (width, height), (0, 255, 0), 2)
            # cv2.line(frame, (0, height), (width, 0), (0, 255, 0), 2)

            # resize the video view
            # frame = cv2.resize(frame, None, fx=0.8, fy=0.8)

            # show the video
            # cv2.imshow("frame", frame)

            # move the video view to the top left corner
            # cv2.moveWindow("frame", 0, 0)

            # quit if q key pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        vid.release()
        cv2.destroyAllWindows()


def main():
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    run()


if __name__ == "__main__":
    main()
