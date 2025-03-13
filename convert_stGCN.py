import numpy as np
import pandas as pd
import json
import os
from glob import glob

# === 행동 클래스 매핑 ===
ACTIVITY_TO_LABEL = {
    "no_activity": 0,
    "walking": 1,
    "standing": 2,
    "sitting": 3,
    "no_presence": 4
}

# === 폴더 생성 함수 ===
def create_folders(*folders):
    """필요한 폴더가 없으면 생성"""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# === CSV 파일 로드 및 변환 함수 ===
def load_skeleton_data(csv_file):
    """CSV 파일을 로드하여 ST-GCN 형식으로 변환"""
    df = pd.read_csv(csv_file, header=None)
    num_frames = df.shape[0]
    total_columns = df.shape[1] - 1  # 첫 번째 열(Frame Index) 제외
    num_joints = total_columns // 2  # 관절 개수 계산 (X, Y)

    print(f"파일: {os.path.basename(csv_file)} - 감지된 관절 개수: {num_joints}")

    # === Skeleton 데이터 변환 (프레임, 관절, 좌표) ===
    skeleton_data = []
    for i in range(num_frames):
        row = df.iloc[i, 1:].values  # 첫 번째 열(Frame) 제외
        joints = np.array(row).reshape(num_joints, 2)  # (관절 개수, 2) 변환
        skeleton_data.append(joints)

    if len(skeleton_data) == 0:
        print(f"🚨 {csv_file} 변환 실패: 유효한 데이터 없음.")
        return None

    # === (Frames, Nodes, Features) → (Batch, Channels, Frames, Nodes) 변환 ===
    skeleton_data = np.array(skeleton_data)  # (Frames, Nodes, Features)
    skeleton_data = np.transpose(skeleton_data, (2, 0, 1))  # (Features, Frames, Nodes)
    skeleton_data = np.expand_dims(skeleton_data, axis=0)  # (Batch, Features, Frames, Nodes)

    return skeleton_data

# === JSON 파일 로드 및 행동 라벨 변환 함수 ===
def load_action_labels(json_file, num_frames):
    """JSON 파일을 로드하여 프레임별 행동 라벨 변환"""
    with open(json_file, "r") as f:
        action_data = json.load(f)

    frame_labels = np.zeros((num_frames,), dtype=int)  # 기본값 0 (no_activity)
    for action in action_data:
        start, end, activity = action["frameRange"][0], action["frameRange"][1], action["activity"]
        frame_labels[start:end + 1] = ACTIVITY_TO_LABEL.get(activity, 0)  # 행동 클래스 적용

    return frame_labels

# === 파일 변환 및 저장 함수 ===
def process_files(input_folder_csv, input_folder_json, output_skeleton_folder, output_labels_folder):
    """CSV + JSON 파일을 ST-GCN 형식으로 변환 및 저장"""
    create_folders(output_skeleton_folder, output_labels_folder)

    csv_files = sorted(glob(os.path.join(input_folder_csv, "*.csv")))
    json_files = sorted(glob(os.path.join(input_folder_json, "*.json")))

    for csv_file, json_file in zip(csv_files, json_files):
        file_name = os.path.basename(csv_file).replace(".csv", "")

        # Skeleton 데이터 변환
        skeleton_data = load_skeleton_data(csv_file)
        if skeleton_data is None:
            continue

        # 행동 라벨 변환
        num_frames = skeleton_data.shape[2]
        frame_labels = load_action_labels(json_file, num_frames)

        # NumPy 파일 저장 (각각 다른 폴더)
        np.save(os.path.join(output_skeleton_folder, f"{file_name}_skeleton.npy"), skeleton_data)
        np.save(os.path.join(output_labels_folder, f"{file_name}_labels.npy"), frame_labels)

        print(f"✅ 변환 완료: {file_name} → 저장 경로: {output_skeleton_folder}, {output_labels_folder}")

    print("🎉 모든 CSV 파일 변환 완료!")

# === 실행 코드 ===
if __name__ == "__main__":
    input_folder_csv = r"C:\Users\User\Desktop\didimdol\csv"
    input_folder_json = r"C:\Users\User\Desktop\didimdol\json"
    output_skeleton_folder = r"C:\Users\User\Desktop\didimdol\skeleton_data"
    output_labels_folder = r"C:\Users\User\Desktop\didimdol\frame_labels"

    process_files(input_folder_csv, input_folder_json, output_skeleton_folder, output_labels_folder)
