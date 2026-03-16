def get_xyc(keypoint):
    x = keypoint["x"] * 256.0
    y = keypoint["y"]* 256.0
    c = keypoint.get("visibility", 1.0)
    return x, y, c

def mid_xyc(keypoint1, keypoint2):
    """Midpoint of two MP landmarks."""
    x1, y1, c1 = get_xyc(keypoint1)
    x2, y2, c2 = get_xyc(keypoint2)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, (c1 + c2) / 2.0

def convert_format(pose_keypoints, multi_hand_keypoints, hand_handedness):
    # -------------------------------------------------------
    # BODY KEYPOINTS — matches OpenPose BODY_25 format
    # -------------------------------------------------------
    # CRITICAL: This mapping MUST match OpenPose BODY_25 joint order (0-24)
    # OpenPose BODY_25 format:
    #   0: Nose, 1: Neck, 2: R Shoulder, 3: R Elbow, 4: R Wrist,
    #   5: L Shoulder, 6: L Elbow, 7: L Wrist, 8: Mid Hip,
    #   9: R Hip, 10: R Knee, 11: R Ankle, 12: L Hip, 13: L Knee, 14: L Ankle,
    #   15: R Eye, 16: L Eye, 17: R Ear, 18: L Ear,
    #   19: L Big Toe, 20: L Small Toe, 21: L Heel,
    #   22: R Big Toe, 23: R Small Toe, 24: R Heel
    #
    # Note: Joints {9-14, 19-24} are EXCLUDED in preprocessing (body_pose_exclude)
    # Final output: 13 body + 21 left hand + 21 right hand = 55 keypoints
    # -------------------------------------------------------

    # convert mediapipe formats to OpenPose BODY_25 format
    # pose_keypoints indices:

    op_body = []
    if pose_keypoints:
        op_body = [
            get_xyc(pose_keypoints[0]),                                   # 0: Nose obtained directly from NOSE mediapipe keypoint
            mid_xyc(pose_keypoints[11], pose_keypoints[12]),              # 1: Neck approximated using LEFT_SHOULDER and RIGHT_SHOULDER mediapipe keypoints
            get_xyc(pose_keypoints[12]),                                  # 2: Right Shoulder obtained directly from RIGHT_SHOULDER mediapipe keypoint
            get_xyc(pose_keypoints[14]),                                  # 3: Right Elbow obtained directly from RIGHT_ELBOW mediapipe keypoint
            get_xyc(pose_keypoints[16]),                                  # 4: Right Wrist obtained directly from RIGHT_WRIST mediapipe keypoint
            get_xyc(pose_keypoints[11]),                                  # 5: Left Shoulder obtained directly from LEFT_SHOULDER mediapipe keypoint
            get_xyc(pose_keypoints[13]),                                  # 6: Left Elbow obtained directly from LEFT_ELBOW mediapipe keypoint
            get_xyc(pose_keypoints[15]),                                  # 7: Left Wrist  obtained directly from LEFT_WRIST mediapipe keypoint
            mid_xyc(pose_keypoints[23], pose_keypoints[24]),              # 8: Mid Hip approximated using LEFT_HIP and RIGHT_HIP mediapipe keypoints 
            (0.0, 0.0, 0.0),                                                    # 9: Originally Right Hip (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 10: Originally Right Knee (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 11: Originally Right Ankle (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 12: Originally Left Hip (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 13: Originally Left Knee (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 14: Originally Left Ankle (EXCLUDED)
            get_xyc(pose_keypoints[5]),                                   # 15: Right Eye obtained directly from RIGHT_EYE mediapipe keypoint
            get_xyc(pose_keypoints[2]),                                   # 16: Left Eye obtained directly from LEFT_EYE mediapipe keypoint
            get_xyc(pose_keypoints[8]),                                   # 17: Right Ear obtained directly from RIGHT_EAR mediapipe keypoint
            get_xyc(pose_keypoints[7]),                                   # 18: Left Ear obtained directly from LEFT_EAR mediapipe keypoint
            (0.0, 0.0, 0.0),                                                    # 19: Originally Left Big Toe (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 20: Originally Left Small Toe (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 21: Originally Left Heel (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 22: Originally Right Big Toe (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 23: Originally Right Small Toe (EXCLUDED)
            (0.0, 0.0, 0.0),                                                    # 24: Originally Right Heel (EXCLUDED)
        ]
    else:
        op_body = [(0.0, 0.0, 0.0)] * 25
    
    # Flatten to [(x1, y1, c1), (x2, y2, c2) ... ] to [x1, y1, c1, x2, y2, c2, ...]
    flat_pose_keypoints = [v for trip in op_body for v in trip]
    
    # -------------------------------------------------------
    # (B) HAND KEYPOINTS — 21 points each, OpenPose format
    # -------------------------------------------------------
    # CRITICAL: MediaPipe and OpenPose both use 21 hand keypoints
    # Order is IDENTICAL: WRIST (0), then 4 points per finger (THUMB, INDEX, MIDDLE, RING, PINKY)
    # MediaPipe hand order: 0=WRIST, 1-4=THUMB, 5-8=INDEX, 9-12=MIDDLE, 13-16=RING, 17-20=PINKY
    # OpenPose hand order:  0=WRIST, 1-4=THUMB, 5-8=INDEX, 9-12=MIDDLE, 13-16=RING, 17-20=PINKY
    # ✓ Direct 1:1 mapping - no reordering needed
    # -------------------------------------------------------
    left_hand = [(0.0, 0.0, 0.0)] * 21
    right_hand = [(0.0, 0.0, 0.0)] * 21

    if multi_hand_keypoints:
        for single_hand_keypoints, hand in zip(multi_hand_keypoints, hand_handedness):        
            # Convert MediaPipe hand landmarks → 256px coords (OpenPose format)
            # MediaPipe landmarks are in normalized [0,1] space
            # We scale to 256x256 to match training data preprocessing
            # Note: Normalization to [-1,1] happens later in preprocessing
            points = []
            for keypoint in single_hand_keypoints:
                x = keypoint["x"] * 256.0
                y = keypoint["y"] * 256.0
                points.append((1.0, 1.0, 1.0))
            
            if hand[0]["categoryName"] == "Left":
                left_hand = points
            elif hand[0]["categoryName"] == "Right":
                right_hand = points
    
    # flatten
    flat_left_hand_keypoints = [v for trip in left_hand for v in trip]
    flat_right_hand_keypoints = [v for trip in right_hand for v in trip]
    
    return flat_pose_keypoints, flat_left_hand_keypoints, flat_right_hand_keypoints