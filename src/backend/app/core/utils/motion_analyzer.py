import numpy as np

def movement_score(landmarks_frames,hand_weight=3):
    hand_indices = list(range(13)) 
    diffs = []
    for i in range(len(landmarks_frames) - 1):
        diff = np.linalg.norm(landmarks_frames[i+1] - landmarks_frames[i], axis=1)  # shape (num_landmarks,)

        # Apply 2x weight to hand landmarks TODO: maybe only put weight on the hands?
        weighted_diff = diff.copy()
        weighted_diff[hand_indices] *= hand_weight

        total_diff = np.sum(weighted_diff)
        diffs.append(total_diff)

    avg_movement = np.mean(diffs)
    return avg_movement

def update_ema(old_ema, new_score, alpha=0.3):
    if old_ema:
        return alpha * new_score + (1 - alpha) * old_ema
    else:
        return new_score