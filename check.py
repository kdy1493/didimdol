import numpy as np
import os

# 상대 경로 설정 (현재 스크립트가 실행되는 위치 기준)
skeleton_folder = "./skeleton_data"
labels_folder = "./frame_labels"

# 불러올 파일 지정 (파일명 수정 가능)
skeleton_file = os.path.join(skeleton_folder, "video_20250225_100541_440_skeleton.npy")
labels_file = os.path.join(labels_folder, "video_20250225_100541_440_labels.npy")

# NumPy 파일 로드 (allow_pickle=True 설정)
frame_labels = np.load(labels_file, allow_pickle=True)
skeleton_data = np.load(skeleton_file, allow_pickle=True)

print("Skeleton 데이터 형태:", skeleton_data.shape)  # (1, 2, Frames, 17)
print("Frame Labels 데이터 형태:", frame_labels.shape)  # (Frames,)
