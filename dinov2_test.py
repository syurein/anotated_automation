import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import shutil
import glob
from icrawler.builtin import BingImageCrawler

# --- 1. 設定 ---

# 学習させたいクラス名と、画像収集に使う検索キーワードを定義
# 'クラス名': '検索キーワード'
TRAINING_KEYWORDS = {
    '散らかっている': 'ごみ　散らかっている',
    '散らかっていない': '空のゴミ箱　ゴミが入っている',
    'other': 'image' # 「その他」クラス用の画像
}

# 定数
N_SHOTS = 20  # 各クラスで収集・学習に使う画像数
CLASS_NAMES = list(TRAINING_KEYWORDS.keys())
NUM_TOTAL_CLASSES = len(CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "dinov2_custom_classifier_head.pth" # 保存するモデルのパス
TRAINING_DATA_DIR = "./training_images" # 学習用画像を保存するディレクトリ

print(f"Using device: {DEVICE}")
print(f"Training for {NUM_TOTAL_CLASSES} classes: {', '.join(CLASS_NAMES)}")


# --- 2. DINOv2モデルとカスタム分類器の定義 ---
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(DINOv2Classifier, self).__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        feature_dim = self.dinov2.embed_dim
        self.linear_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.dinov2(x)
        outputs = self.linear_head(features)
        return outputs

# --- 3. 画像収集とデータセット準備 ---

# DINOv2用の画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def crawl_images_for_training():
    """
    学習用の画像をicrawlerで収集する関数。
    既存のデータがある場合はユーザーに入力を求める。
    """
    print("\n--- Preparing/Crawling images for training ---")
    
    perform_crawl = True
    # 学習データディレクトリが存在する場合の処理
    if os.path.exists(TRAINING_DATA_DIR):
        while True:
            # ユーザーに行動を選択させる
            choice = input(
                f"Training data directory '{TRAINING_DATA_DIR}' already exists. \n"
                "Choose an action - [add/reset/skip]: "
            ).lower().strip()

            if choice == 'reset':
                print(f"Resetting: Removing old training data directory...")
                shutil.rmtree(TRAINING_DATA_DIR)
                break 
            elif choice == 'add':
                print("Adding: New images will be added to the existing directory.")
                break
            elif choice == 'skip':
                print("Skipping crawl. Using existing data for training.")
                perform_crawl = False
                break
            else:
                print("Invalid input. Please enter 'add', 'reset', or 'skip'.")
    
    # 'reset'または'add'を選択した場合、あるいはディレクトリが存在しなかった場合にクロールを実行
    if perform_crawl:
        print("\n--- Crawling images ---")
        for class_name, keyword in TRAINING_KEYWORDS.items():
            class_dir = os.path.join(TRAINING_DATA_DIR, class_name)
            print(f"Crawling for '{class_name}' with keyword '{keyword}'...")
            crawler = BingImageCrawler(storage={'root_dir': class_dir})
            # N_SHOTS枚の画像を収集
            crawler.crawl(keyword=keyword, max_num=N_SHOTS)
        print("Finished crawling training images.")

    # 既存データを使用する場合も、クロールした場合も、最終的にデータセットを作成する
    try:
        train_dataset = datasets.ImageFolder(TRAINING_DATA_DIR, transform=transform)
        if len(train_dataset) == 0:
            print("Error: No images found in the training directory. Please run again and choose 'reset' or 'add'.")
            exit() # 画像がない場合はプログラムを終了
        print(f"\nDataset created. Found {len(train_dataset)} images in total.")
        print(f"Class mapping: {train_dataset.class_to_idx}")
        return train_dataset
    except FileNotFoundError:
        print(f"Error: Training directory '{TRAINING_DATA_DIR}' not found and crawling was skipped. No data to train on.")
        exit()


# --- 4. 線形層の学習とモデル保存 ---

def train_and_save_model(support_dataset):
    """モデルを学習し、重みを保存する関数"""
    print("\n--- Training Linear Head ---")
    
    model = DINOv2Classifier(num_classes=NUM_TOTAL_CLASSES).to(DEVICE)
    support_loader = DataLoader(support_dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.linear_head.parameters(), lr=1e-3)
    
    num_epochs = 40 
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in support_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(support_loader):.4f}")

    print("Finished Training.")
    
    torch.save(model.linear_head.state_dict(), MODEL_SAVE_PATH)
    print(f"Model's linear head saved to {MODEL_SAVE_PATH}")
    return support_dataset.class_to_idx

# --- 5. 収集した画像の分類（推論） ---

def crawl_and_predict_images(class_to_idx):
    """推論用の新しい画像を収集し、分類する関数"""
    print("\n--- Crawling new images for prediction ---")
    
    prediction_keywords = {
    'ごみ　散らかっている': 10,
    'ゴミ箱　空': 10,
    'image': 5  # 「その他」クラス用の画像
    }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = DINOv2Classifier(num_classes=NUM_TOTAL_CLASSES).to(DEVICE)
    model.linear_head.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    for keyword, num_images in prediction_keywords.items():
        image_dir = os.path.join('./prediction_images', keyword.replace(" ", "_"))
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)

        print(f"\n--- Predicting for keyword: '{keyword}' ---")
        crawler = BingImageCrawler(storage={'root_dir': image_dir})
        crawler.crawl(keyword=keyword, max_num=num_images)

        image_paths = glob.glob(os.path.join(image_dir, '*'))
        if not image_paths:
            print("No images found for prediction.")
            continue
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    
                    outputs = model(image_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    predicted_class_name = idx_to_class[predicted_idx.item()]
                    
                    print(f"Image: {os.path.basename(img_path):<20} -> Predicted: {predicted_class_name}")

                except Exception as e:
                    print(f"Could not process {os.path.basename(img_path)}. Error: {e}")


# --- メイン実行部分 ---
if __name__ == '__main__':
    train_dataset = crawl_images_for_training()
    class_map = train_and_save_model(train_dataset)
    crawl_and_predict_images(class_map)
