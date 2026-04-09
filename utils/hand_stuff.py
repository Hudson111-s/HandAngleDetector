import cv2
from utils.hand_enums import HandKeypoints

def get_hand_keypoint(hand_landmarks: list) -> dict:
    hand_keypoint_dict = {}
    for index, hl in enumerate(hand_landmarks):
        hand_keypoint_dict[HandKeypoints(index)] = hl

    return hand_keypoint_dict


def hand_keypoint_to_pixel_c(hand_keypoint_dict: dict, w: int, h: int) -> dict:
    hand_keypoint_pixel_c = hand_keypoint_dict.copy()
    for key, hl in hand_keypoint_dict.items():
        hand_keypoint_pixel_c[key] = (int(hl.x * w), int(hl.y * h))

    return hand_keypoint_pixel_c


def draw_circle(frame, ploc: tuple) -> None:
    cv2.circle(frame, ploc, 8, (0, 255, 0), 1)


def draw_line(frame, ploc: tuple, a: tuple) -> None:
    cv2.line(frame, ploc, a, (255, 0, 0), 2)
