"""
MediaPipe pose extraction from video frames.
Extracts keypoints in OpenPose format for ASL recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
mp_pose_mod = mp.solutions.pose
mp_hands_mod = mp.solutions.hands



class PoseExtractor:
    """Extract pose keypoints from video frames using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe solutions."""
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False
        )
        
        # Initialize hand detectors
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def extract_keypoints(self, frame):
        """
        Extract keypoints from a single frame in TRUE OpenPose format.
        
        CRITICAL REQUIREMENTS FOR TGCN MODEL COMPATIBILITY:
        ===================================================
        1. MediaPipe → OpenPose BODY_25 joint mapping (0-24) - EXACT order required
        2. Hand keypoints: 21 points each (MediaPipe order = OpenPose order)
        3. Coordinates scaled to 256x256 space (normalization to [-1,1] done in preprocessing)
        4. Output format: OpenPose JSON structure
        
        MEDIAPIPE → OPENPOSE BODY_25 MAPPING TABLE:
        ===========================================
        OpenPose Index | OpenPose Name      | MediaPipe Landmark              | Notes
        -------------- | ------------------ | ------------------------------- | -----
        0              | Nose               | NOSE                           | Direct
        1              | Neck               | (LEFT_SHOULDER + RIGHT_SHOULDER)/2 | Midpoint
        2              | R Shoulder         | RIGHT_SHOULDER                  | Direct
        3              | R Elbow            | RIGHT_ELBOW                      | Direct
        4              | R Wrist            | RIGHT_WRIST                      | Direct
        5              | L Shoulder         | LEFT_SHOULDER                    | Direct
        6              | L Elbow            | LEFT_ELBOW                       | Direct
        7              | L Wrist            | LEFT_WRIST                       | Direct
        8              | Mid Hip            | (LEFT_HIP + RIGHT_HIP)/2         | Midpoint
        9              | R Hip              | RIGHT_HIP                        | EXCLUDED in preprocessing
        10             | R Knee             | RIGHT_KNEE                       | EXCLUDED in preprocessing
        11             | R Ankle            | RIGHT_ANKLE                      | EXCLUDED in preprocessing
        12             | L Hip              | LEFT_HIP                         | EXCLUDED in preprocessing
        13             | L Knee             | LEFT_KNEE                        | EXCLUDED in preprocessing
        14             | L Ankle            | LEFT_ANKLE                       | EXCLUDED in preprocessing
        15             | R Eye              | RIGHT_EYE                        | Direct
        16             | L Eye              | LEFT_EYE                         | Direct
        17             | R Ear              | RIGHT_EAR                        | Direct
        18             | L Ear              | LEFT_EAR                         | Direct
        19             | L Big Toe          | LEFT_FOOT_INDEX                  | EXCLUDED in preprocessing
        20             | L Small Toe        | Approximated from ankle+foot_index | EXCLUDED in preprocessing
        21             | L Heel             | LEFT_HEEL                        | EXCLUDED in preprocessing
        22             | R Big Toe          | RIGHT_FOOT_INDEX                 | EXCLUDED in preprocessing
        23             | R Small Toe        | Approximated from ankle+foot_index | EXCLUDED in preprocessing
        24             | R Heel             | RIGHT_HEEL                       | EXCLUDED in preprocessing
        
        FINAL OUTPUT AFTER PREPROCESSING:
        =================================
        - Body keypoints included: 0-8, 15-18 (13 keypoints)
        - Body keypoints excluded: 9-14, 19-24 (12 keypoints)
        - Left hand: 21 keypoints (all included)
        - Right hand: 21 keypoints (all included)
        - TOTAL: 55 keypoints (matches NUM_NODES)
        
        NORMALIZATION (done in preprocessing, not here):
        ================================================
        Coordinates are scaled to 256x256 here, then normalized in preprocessing:
        norm_x = 2 * ((x / 256.0) - 0.5)  → range [-1, 1]
        norm_y = 2 * ((y / 256.0) - 0.5)  → range [-1, 1]
        """

        # Convert frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Run MediaPipe
        pose_results = self.pose.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)

        # --------------------------
        # Helper functions
        # --------------------------
        def get_xyc(lm):
            """Convert a MediaPipe landmark to 256x256 coordinate space."""
            x = lm.x * 256.0
            y = lm.y * 256.0
            c = lm.visibility if hasattr(lm, "visibility") else 1.0
            return x, y, c

        def mid_xyc(lm1, lm2):
            """Midpoint of two MP landmarks."""
            x1, y1, c1 = get_xyc(lm1)
            x2, y2, c2 = get_xyc(lm2)
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0, (c1 + c2) / 2.0

        def approx_small_toe(ankle_lm, foot_index_lm):
            """
            Approximate small toe position from ankle and foot_index.
            MediaPipe doesn't have a separate small toe landmark.
            We approximate it by offsetting from foot_index towards the ankle.
            """
            x_ankle, y_ankle, c_ankle = get_xyc(ankle_lm)
            x_index, y_index, c_index = get_xyc(foot_index_lm)
            # Small toe is approximately 0.3 of the way from foot_index towards ankle
            x_small = x_index + 0.3 * (x_ankle - x_index)
            y_small = y_index + 0.3 * (y_ankle - y_index)
            c_small = (c_ankle + c_index) / 2.0
            return x_small, y_small, c_small

        # -------------------------------------------------------
        # (A) BODY KEYPOINTS — mapped to OpenPose BODY_25 format
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
        op_body = []  # will store 25 entries of (x,y,c)

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            PL = self.mp_pose.PoseLandmark

            # Build in EXACT OpenPose BODY_25 order (0–24)
            op_body = [
                get_xyc(lm[PL.NOSE.value]),                                   # 0: Nose
                mid_xyc(lm[PL.LEFT_SHOULDER.value], lm[PL.RIGHT_SHOULDER.value]),  # 1: Neck (midpoint)
                get_xyc(lm[PL.RIGHT_SHOULDER.value]),                         # 2: R Shoulder
                get_xyc(lm[PL.RIGHT_ELBOW.value]),                            # 3: R Elbow
                get_xyc(lm[PL.RIGHT_WRIST.value]),                            # 4: R Wrist
                get_xyc(lm[PL.LEFT_SHOULDER.value]),                          # 5: L Shoulder
                get_xyc(lm[PL.LEFT_ELBOW.value]),                             # 6: L Elbow
                get_xyc(lm[PL.LEFT_WRIST.value]),                             # 7: L Wrist
                mid_xyc(lm[PL.LEFT_HIP.value], lm[PL.RIGHT_HIP.value]),       # 8: Mid Hip (midpoint)
                get_xyc(lm[PL.RIGHT_HIP.value]),                              # 9: R Hip (EXCLUDED)
                get_xyc(lm[PL.RIGHT_KNEE.value]),                             # 10: R Knee (EXCLUDED)
                get_xyc(lm[PL.RIGHT_ANKLE.value]),                            # 11: R Ankle (EXCLUDED)
                get_xyc(lm[PL.LEFT_HIP.value]),                               # 12: L Hip (EXCLUDED)
                get_xyc(lm[PL.LEFT_KNEE.value]),                              # 13: L Knee (EXCLUDED)
                get_xyc(lm[PL.LEFT_ANKLE.value]),                             # 14: L Ankle (EXCLUDED)
                get_xyc(lm[PL.RIGHT_EYE.value]),                              # 15: R Eye
                get_xyc(lm[PL.LEFT_EYE.value]),                               # 16: L Eye
                get_xyc(lm[PL.RIGHT_EAR.value]),                              # 17: R Ear
                get_xyc(lm[PL.LEFT_EAR.value]),                               # 18: L Ear
                get_xyc(lm[PL.LEFT_FOOT_INDEX.value]),                        # 19: L Big Toe (EXCLUDED)
                approx_small_toe(lm[PL.LEFT_ANKLE.value], lm[PL.LEFT_FOOT_INDEX.value]),  # 20: L Small Toe (EXCLUDED, approximated)
                get_xyc(lm[PL.LEFT_HEEL.value]),                              # 21: L Heel (EXCLUDED)
                get_xyc(lm[PL.RIGHT_FOOT_INDEX.value]),                       # 22: R Big Toe (EXCLUDED)
                approx_small_toe(lm[PL.RIGHT_ANKLE.value], lm[PL.RIGHT_FOOT_INDEX.value]),  # 23: R Small Toe (EXCLUDED, approximated)
                get_xyc(lm[PL.RIGHT_HEEL.value]),                             # 24: R Heel (EXCLUDED)
            ]
        else:
            op_body = [(0.0, 0.0, 0.0)] * 25

        # Flatten to [x,y,c,x,y,c,...]
        pose_keypoints = [v for trip in op_body for v in trip]

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

        if hand_results.multi_hand_landmarks:
            for hand_lm, handed in zip(hand_results.multi_hand_landmarks,
                                    hand_results.multi_handedness):

                label = handed.classification[0].label

                # Convert MediaPipe hand landmarks → 256px coords (OpenPose format)
                # MediaPipe landmarks are in normalized [0,1] space
                # We scale to 256x256 to match training data preprocessing
                # Note: Normalization to [-1,1] happens later in preprocessing
                pts = []
                for lm in hand_lm.landmark:
                    x = lm.x * 256.0
                    y = lm.y * 256.0
                    pts.append((x, y, 1.0))

                if label == "Left":
                    left_hand = pts
                elif label == "Right":
                    right_hand = pts

        # flatten
        left_hand_keypoints = [v for trip in left_hand for v in trip]
        right_hand_keypoints = [v for trip in right_hand for v in trip]

        # -------------------------------------------------------
        # (C) BUILD OpenPose-style JSON
        # -------------------------------------------------------
        keypoint_data = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": pose_keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": left_hand_keypoints,
                "hand_right_keypoints_2d": right_hand_keypoints,
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        return keypoint_data
    
    def extract_from_video(self, video_path, max_frames=None, sample_rate=1):
        """
        Extract keypoints from all frames in a video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None for all)
            sample_rate: Extract every Nth frame (1 = all frames)
            
        Returns:
            list: List of keypoint dictionaries (one per frame)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames_data = []
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting keypoints from video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_count % sample_rate == 0:
                keypoints = self.extract_keypoints(frame)
                frames_data.append(keypoints)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted keypoints from {extracted_count} frames")
        
        return frames_data
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()

