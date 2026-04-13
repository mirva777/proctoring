import numpy as np

from detectors.pose_gaze_estimator import HeadPoseEstimator


def test_normalize_pose_angles_folds_180_degree_yaw_back_to_frontal_range():
    yaw, pitch, roll = HeadPoseEstimator.normalize_pose_angles(177.47, 7.71, 13.86)
    assert round(yaw, 2) == -2.53
    assert round(pitch, 2) == 7.71
    assert round(roll, 2) == 13.86


def test_normalize_pose_angles_handles_negative_180_degree_yaw():
    yaw, pitch, roll = HeadPoseEstimator.normalize_pose_angles(-179.64, 7.38, 10.60)
    assert round(yaw, 2) == 0.36
    assert round(pitch, 2) == 7.38
    assert round(roll, 2) == 10.60


def test_classify_attention_marks_centered_face_as_clean():
    estimator = HeadPoseEstimator()
    look_away, severity, direction = estimator.classify_attention(-2.53, 7.71)
    assert look_away is False
    assert severity == "none"
    assert direction == "center"


def test_classify_attention_marks_large_yaw_as_severe():
    estimator = HeadPoseEstimator()
    look_away, severity, direction = estimator.classify_attention(42.0, 3.0)
    assert look_away is True
    assert severity == "severe"
    assert direction == "right"


def test_pose_from_face_transform_reads_identity_matrix_as_centered_pose():
    yaw, pitch, roll = HeadPoseEstimator.pose_from_face_transform(np.eye(4)) or (None, None, None)
    assert round(yaw or 0.0, 2) == 0.0
    assert round(pitch or 0.0, 2) == 0.0
    assert round(roll or 0.0, 2) == 0.0
