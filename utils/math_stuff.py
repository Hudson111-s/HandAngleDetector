import math

def get_midpoint(p1: tuple, p2: tuple) -> tuple:
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


def get_angle_from_2lines(l1: tuple, l2: tuple, a: tuple):
    ang1 = math.atan2(l1[1] - a[1], l1[0] - a[0])
    ang2 = math.atan2(l2[1] - a[1], l2[0] - a[0])
    angle = abs(math.degrees(ang1 - ang2))
    if angle > 180: angle = 360 - angle

    return angle


def get_all_finger_angles(hand_keypoint_pixel_c: dict) -> list:
    finger_angles = []
    for i in range(4): # All fingers - 1
        k = hand_keypoint_pixel_c[(i * 4) + 1]
        other_k = hand_keypoint_pixel_c[(4 * i) + 5]
        midpoint = get_midpoint(k, other_k)

        t = hand_keypoint_pixel_c[(4 * i) + 4]
        other_t = hand_keypoint_pixel_c[(4 * i) + 8]
        angle = get_angle_from_2lines(t, other_t, midpoint)

        finger_angles.append((angle, (t, other_t, midpoint)))

    return finger_angles


        




