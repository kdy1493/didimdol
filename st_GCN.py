import torch
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from stgcn_graph import STGCNGraphConv  # STGCNGraphConv 불러오기
from convert_stGCN import ACTIVITY_TO_LABEL  # 행동 클래스 매핑 딕셔너리

# === CUDA 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# === 데이터 폴더 설정 ===
skeleton_folder = "./skeleton_data"
labels_folder = "./frame_labels"

# === 파일 리스트 가져오기 ===
skeleton_files = sorted(glob(os.path.join(skeleton_folder, "*.npy")))
label_files = sorted(glob(os.path.join(labels_folder, "*.npy")))

# === PyTorch Dataset 정의 ===
class SkeletonDataset(Dataset):
    def __init__(self, skeleton_files, label_files):
        self.skeleton_files = skeleton_files
        self.label_files = label_files

    def __len__(self):
        return len(self.skeleton_files)

    def __getitem__(self, idx):
        # NumPy 데이터 로드
        skeleton_data = np.load(self.skeleton_files[idx], allow_pickle=True)  # (1, 2, Frames, Nodes)
        frame_labels = np.load(self.label_files[idx], allow_pickle=True)  # (Frames,)

        # PyTorch Tensor 변환 (GPU로 이동)
        skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(frame_labels, dtype=torch.long).unsqueeze(0).to(device)  # (1, Frames)

        return skeleton_tensor, labels_tensor

# === K-Fold Cross Validation 설정 ===
num_folds = 5
num_epochs = 200
batch_size = 16
patience = 10  # Early Stopping 기준 (10 Epoch 동안 개선 없으면 종료)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# === Dataset 로드 ===
dataset = SkeletonDataset(skeleton_files, label_files)

# === Cross Validation 실행 ===
test_losses = []
test_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n🚀 Fold [{fold+1}/{num_folds}] 시작!")

    # Train / Validation Subset 생성
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # DataLoader 설정
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # === STGCNGraphConv 모델 초기화 (GPU로 이동) ===
    num_classes = len(ACTIVITY_TO_LABEL)  # 행동 클래스 개수
    args = {
        "Kt": 3, "Ks": 3, "n_his": 10, "act_func": "relu", 
        "graph_conv_type": "gcn", "gso": None, "enable_bias": True, "droprate": 0.3
    }
    blocks = [[2, 64], [64, 128], [128, 256], [256, num_classes]]  # STGCN 블록 구조
    model = STGCNGraphConv(args, blocks, n_vertex=17).to(device)

    # 학습 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Early Stopping 설정
    best_val_loss = float("inf")
    patience_counter = 0

    # === 학습 루프 ===
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0

        # === Training ===
        model.train()
        for batch_skeleton, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_skeleton)  # (Batch, Classes, Frames)
            
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # === Validation ===
        model.eval()
        with torch.no_grad():
            for batch_skeleton, batch_labels in val_loader:
                outputs = model(batch_skeleton)
                loss = criterion(outputs.squeeze(), batch_labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Fold [{fold+1}/{num_folds}], Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # === Early Stopping 체크 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 개선되었으므로 초기화
        else:
            patience_counter += 1  # 개선되지 않았으므로 카운트 증가

        if patience_counter >= patience:
            print(f"⏹ Early Stopping! Epoch [{epoch+1}/{num_epochs}]에서 종료 (Val Loss 개선 없음)")
            break

    print(f"✅ Fold [{fold+1}/{num_folds}] 완료!")

    # === Test Set에서 최종 성능 평가 ===
    test_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loss = 0
    correct_preds = 0
    total_preds = 0

    model.eval()
    with torch.no_grad():
        for batch_skeleton, batch_labels in test_loader:
            outputs = model(batch_skeleton)
            loss = criterion(outputs.squeeze(), batch_labels)
            test_loss += loss.item()

            # Accuracy 계산 (예측값과 실제 라벨 비교)
            predicted = torch.argmax(outputs, dim=1)
            correct_preds += (predicted == batch_labels).sum().item()
            total_preds += batch_labels.numel()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_preds / total_preds

    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)

    print(f"📊 Fold [{fold+1}/{num_folds}] Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# === Cross Validation 결과 출력 ===
print("\n🎯 최종 Test 결과:")
print(f"📉 평균 Test Loss: {np.mean(test_losses):.4f}")
print(f"✅ 평균 Test Accuracy: {np.mean(test_accuracies):.4f}")
