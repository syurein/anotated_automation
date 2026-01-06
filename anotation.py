import os
import json
import random
import requests
import cv2

from pycocotools.coco import COCO  # インストール済みであることが前提
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForZeroShotObjectDetection
import numpy as np
import subprocess
from tqdm import tqdm
import zipfile
import moondream as md
# 環境検出
try:
    from google.colab import userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# 環境に応じたインポート
if IS_COLAB:
    from google.colab.patches import cv2_imshow
    from google import genai
else:
    # ローカル環境用の代替処理
    def cv2_imshow(image):
        """ローカル環境での画像表示関数"""
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # ローカル環境ではgenaiを別途インストール必要
    try:
        from google import genai
    except ImportError:
        genai = None
        print("ローカル環境ではGoogle GenAIは利用できません")








def compute_iou(boxA, boxB):
    """
    boxA, boxB: [ymin, xmin, ymax, xmax] 形式
    """
    yminA, xminA, ymaxA, xmaxA = boxA
    yminB, xminB, ymaxB, xmaxB = boxB

    inter_xmin = max(xminA, xminB)
    inter_ymin = max(yminA, yminB)
    inter_xmax = min(xmaxA, xmaxB)
    inter_ymax = min(ymaxA, ymaxB)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    areaA = (xmaxA - xminA) * (ymaxA - yminA)
    areaB = (xmaxB - xminB) * (ymaxB - yminB)

    iou = inter_area / float(areaA + areaB - inter_area + 1e-6)
    return iou

def _convert_to_yolo_format(box_2d):
    """
    [ymin, xmin, ymax, xmax] 形式の正規化座標を
    YOLO 形式 (<x_center> <y_center> <width> <height>) に変換する。
    """
    ymin, xmin, ymax, xmax = box_2d
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + (w / 2)
    y_center = ymin + (h / 2)
    return x_center, y_center, w, h
#YOLO形式への変換



class COCOHandler:
    """
    COCO データセットの画像・アノテーション取得を扱うクラス。
    """
    def __init__(self, data_dir='coco_data', annotation_url=None, annotation_path=None, dataset_type='val2017'):
        """
        :param data_dir: 画像/アノテーション保存ディレクトリ
        :param annotation_url: JSON をダウンロードする URL（省略可）
        :param annotation_path: 既にダウンロード済み JSON のパス（どちらか必須）
        :param dataset_type: 'train2017' など
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, dataset_type)
        os.makedirs(self.images_dir, exist_ok=True)

        # アノテーションファイルの取得
        if annotation_path:
            ann_file = annotation_path
        else:
            if annotation_url is None:
                raise ValueError("annotation_url か annotation_path を指定してください")
            os.makedirs(data_dir, exist_ok=True)
            ann_filename = os.path.basename(annotation_url)
            ann_file = os.path.join(data_dir, ann_filename)
            if not os.path.exists(ann_file):
                print(f"Downloading annotation file from {annotation_url} ...")
                resp = requests.get(annotation_url, stream=True)
                resp.raise_for_status()
                with open(ann_file, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("ダウンロード完了。")
        # COCO API をロード
        self.coco = COCO(ann_file)

    def get_image_ids(self, category_name):
        # カテゴリ名から画像 ID のリストを取得
        cat_ids = self.coco.getCatIds(catNms=[category_name])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        return img_ids

    def download_image(self, img_info, force=False):
        """
        1 画像をダウンロードしてローカルに保存。
        :param img_info: COCO の img 情報 dict
        :param force: 上書きダウンロードするか
        :return: 保存パス
        """
        img_id = img_info['id']
        file_name = img_info['file_name']
        url = img_info.get('coco_url')
        save_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(save_path) or force:
            print(f"Downloading image {img_id} ...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return save_path

class KosmosInference:

    def __init__(self,model_id="microsoft/kosmos-2-patch14-224",device=None):
      self.ckpt = model_id
      self.device="cuda" if torch.cuda.is_available() else "cpu"
      self.model = AutoModelForVision2Seq.from_pretrained(self.ckpt).to(self.device)
      self.processor = AutoProcessor.from_pretrained(self.ckpt)

    def get_response(self,image_path,prompt="<grounding>An image of"):
        if prompt in 'all':
          prompt=prompt
        else:
          prompt = f"<grounding>An image of"
        print(f'Kosmos:{prompt}')
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
          text=prompt,
          images=image,
          return_tensors="pt"
        ).to(self.device)
        # --- 推論 ---
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=256,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        tmp, entities = self.processor.post_process_generation(generated_text)
        print(f"\n=== Detected Entities ===")
        for entity in entities:
            entity_name, _, bboxes = entity

            print(f"- {entity_name} | bboxes: {bboxes}")
        return entities

    #[('a cat ears', (35, 45), [(0.296875, 0.046875, 0.390625, 0.203125), (0.546875, 0.046875, 0.609375, 0.140625)]), ('a maid outfit', (50, 63), [(0.234375, 0.046875, 0.640625, 0.828125)])]


    def parse_response(self, resp_text):
        """
        get_response で返した JSON文字列をパースし、
        Gemini と同じ形式の list[dict] に揃える
        """
        parsed=[]
        for entity in resp_text:
          entity_name, _, bboxes = entity
          for box in bboxes:
            parsed.append({
                   "label":entity_name,
                   "box_2d":[
                       box[0],box[1],box[2],box[3]
                   ]})
        print(parsed)
        return parsed
    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        指定されたフォルダ内の全画像に対して推論を行い、YOLO形式の学習データを作成する。
        内部で自身の get_response と parse_response を呼び出す。
        """
        os.makedirs(output_folder_path, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt == None:
            prompt = (f"Detect all prominent items from the following list: {class_names} in the image. "
                      "The response should be a JSON array. Each object should have a 'label' and 'box_2d'. "
                      "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.")
           

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # 自身の推論メソッドを呼び出す
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')

                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")


import os
import time
import json
import requests
from PIL import Image
from tqdm import tqdm


class DINOXInference:
    def __init__(self, api_token, device=None):
        """
        DINO-X API用の推論クラス
        :param api_token: DINO-X Platformのアクセストークン
        """
        self.api_token = api_token
        self.base_url = "https://api.deepdataspace.com/v2"
        self.headers = {
            "Token": self.api_token
        }
        # API版ではローカルのdevice(cuda/cpu)は使用しませんが、インターフェース維持のため保持
        self.device = device 

    def get_response(self, image_path, prompt):
        """
        DINO-X APIを呼び出し、タスクが完了するまで待機して結果を返す
        """
        # 画像サイズ取得（正規化用）
        with Image.open(image_path) as img:
            self.width, self.height = img.size

        # --- Step 1: タスク作成 ---
        # APIのパスはドキュメントに基づき 'dinox/detection' とします
        task_url = f"{self.base_url}/task/dinox/detection"
        
        # ※ファイルアップロードが必要なため、multipart/form-dataで送信します
        # プロンプトなどのパラメータは 'data' フィールドに含めます
        payload = {
            "text_prompt": prompt,
        }
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            resp = requests.post(task_url, headers=self.headers, data=payload, files=files)
        
        if resp.status_code != 200:
            raise Exception(f"Task creation failed: {resp.text}")
            
        json_resp = resp.json()
        if json_resp.get("code") != 0:
            raise Exception(f"API Error: {json_resp.get('msg')}")
            
        task_uuid = json_resp["data"]["task_uuid"]

        # --- Step 2: ポーリング (結果が success か failed になるまで待機) ---
        status_url = f"{self.base_url}/task_status/{task_uuid}"
        
        while True:
            status_resp = requests.get(status_url, headers=self.headers)
            status_data = status_resp.json()
            
            if status_data.get("code") != 0:
                raise Exception(f"Status check failed: {status_data.get('msg')}")
                
            status = status_data["data"]["status"]
            
            if status == "success":
                # 成功したら結果を返す
                return status_data["data"]["result"]
            elif status == "failed":
                raise Exception(f"Task failed: {status_data['data'].get('error')}")
            
            # 1秒待機して再試行
            time.sleep(1)

    def parse_response(self, resp):
        """
        APIのレスポンスを共通フォーマット [ymin, xmin, ymax, xmax] (0-1) に変換
        """
        parsed = []
        # DINO-X APIの一般的なレスポンス形式を想定（objectsキー配下にリスト）
        # ※実際のAPIのJSON構造に合わせて調整が必要な場合があります
        objects = resp.get("objects", [])

        for obj in objects:
            label = obj.get("category", "")
            # APIが返す座標系が [xmin, ymin, xmax, ymax] のピクセル値であると仮定
            bbox = obj.get("bbox", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = bbox

            # === 必ず 0〜1 に正規化 ===
            ymin_norm = ymin / self.height
            xmin_norm = xmin / self.width
            ymax_norm = ymax / self.height
            xmax_norm = xmax / self.width

            parsed.append({
                "label": label,
                "box_2d": [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
            })
        
        return parsed

    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        指定されたフォルダ内の全画像に対してAPI推論を行い、YOLO形式の学習データを作成する
        """
        os.makedirs(output_folder_path, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt is None:
            # DINO-Xはより自然な英語を理解するため、プロンプトをシンプルに構成
            prompt = f"Detect {class_names}."

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # API推論を実行
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')

                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")









class GroundingDINOHFInference:

    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=None):
        self.device='cuda' if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def get_response(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        # 文末にピリオドが推奨されます :contentReference[oaicite:1]{index=1}
        if prompt in 'all':
          
          text = f"{prompt[prompt.find('all ')+3:prompt.find(' in')]}."#特殊パターンのプロンプトのみここで対応
        else:
          text=prompt #普通パターン
        #ここが逆だった
        print(text)
        self.width,self.height=image.size # PIL Image.size returns (width, height)
        inputs = self.processor(images=image, text=text, return_tensors="pt",max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            #threshold=0.3,
            target_sizes=[image.size[::-1]] # target_sizes expects (height, width)
        )
        # 複数返るが今回は1枚画像なので最初のリストを返します
        print(results)
        return results[0]

    def parse_response(self, resp):
      parsed = []
      for box, score, label in zip(resp["boxes"], resp["scores"], resp["labels"]):
          xmin, ymin, xmax, ymax = box.tolist()

          # === 必ず 0〜1 に正規化 ===
          ymin_norm = ymin / self.height
          xmin_norm = xmin / self.width
          ymax_norm = ymax / self.height
          xmax_norm = xmax / self.width

          parsed.append({
              "label": label,
              "box_2d": [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
          })
          print(parsed)
      return parsed
    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        指定されたフォルダ内の全画像に対して推論を行い、YOLO形式の学習データを作成する。
        内部で自身の get_response と parse_response を呼び出す。
        """
        os.makedirs(output_folder_path, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt == None:
            prompt = (f"Detect all prominent items from the following list: {class_names} in the image. "
                      "The response should be a JSON array. Each object should have a 'label' and 'box_2d'. "
                      "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.")
           

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # 自身の推論メソッドを呼び出す
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')

                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")

import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# --- Moondream3Inference クラスの定義 ---
import torch
import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoModelForCausalLM

def _convert_to_yolo_format(box_2d):
    """
    YOLO形式への変換ヘルパー関数
    box_2d: [ymin, xmin, ymax, xmax] (0.0 to 1.0)
    returns: x_center, y_center, width, height (0.0 to 1.0)
    """
    ymin, xmin, ymax, xmax = box_2d
    
    # 0.0〜1.0の範囲にクリップ
    ymin = max(0, min(1, ymin))
    xmin = max(0, min(1, xmin))
    ymax = max(0, min(1, ymax))
    xmax = max(0, min(1, xmax))

    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + (w / 2)
    y_center = ymin + (h / 2)
    
    return x_center, y_center, w, h

import torch
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

class Moondream3Inference:
    def __init__(self, api_key=None):
        # デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device} (Full Precision Mode)")

        # モデルの読み込み (量子化なし / float16)
        print("Moondream3モデルを読み込んでいます...")
        model_id = "moondream/moondream3-preview"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": self.device} if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # モデルに紐付いているtokenizerを参照
        if hasattr(self.model, "tokenizer"):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = None

        # GPUパワーがあるため、コンパイルも試行（推論を高速化）
        try:
            if torch.cuda.is_available():
                print("モデルをコンパイルして推論を最適化しています...")
                self.model.compile()
        except Exception as e:
            print(f"コンパイルをスキップしました (非対応環境など): {e}")

    def get_response(self, image_path, prompt):
        """
        アンサンブル評価スクリプトから呼び出されるメソッド。
        推論結果をJSON文字列で返します。
        """
        image = Image.open(image_path).convert('RGB')
        
        # プロンプトから検出対象を抽出（例: "Detect all persons." -> "person"）
        target = "person" if "person" in prompt.lower() else prompt
        
        # 推論実行
        with torch.inference_mode():
            result = self.model.detect(image, target)
        
        objects = result.get("objects", [])
        
        # 各オブジェクトにラベル名を付与（parse_responseで使用）
        for obj in objects:
            obj["label"] = target.lower()

        return json.dumps(objects)

    def parse_response(self, resp_text):
        """
        get_response のJSON出力を受け取り、共通フォーマットに変換。
        出力形式: [{'label': str, 'box_2d': [ymin, xmin, ymax, xmax]}] (0.0 - 1.0)
        """
        detections = json.loads(resp_text)
        parsed = []
        for obj in detections:
            # Moondream3: x_min, y_min, x_max, y_max
            # 評価スクリプト期待形式: [ymin, xmin, ymax, xmax]
            parsed.append({
                "label": obj.get("label", "person"),
                "box_2d": [
                    obj["y_min"], obj["x_min"],
                    obj["y_max"], obj["x_max"]
                ]
            })
        return parsed

    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        フォルダ内の画像を処理してYOLO形式の学習データを作成する独立メソッド
        """
        os.makedirs(output_folder_path, exist_ok=True)
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        print(f"--- YOLOデータセット作成開始 (Moondream3) ---")
        for filename in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # 既存のロジックを再利用
                raw = self.get_response(image_path, prompt or "person")
                detections = self.parse_response(raw)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        # 外部で定義されている _convert_to_yolo_format を利用
                        xc, yc, w, h = _convert_to_yolo_format(det['box_2d'])
                        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
            except Exception as e:
                print(f"エラー: {filename} の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
class MoondreamInference:
    def __init__(self, api_key=None):
        # デバイスの設定（GPUが利用可能ならGPUを使う）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")

        # モデルとトークナイザーの読み込み
        model_id = "vikhyatk/moondream2"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision="2025-06-21",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def get_response(self, image_path, prompt):
        """画像とプロンプトから物体検出結果を取得"""
        image = Image.open(image_path).convert('RGB')
        result = self.model.detect(image, prompt)
        return json.dumps(result["objects"])

    def parse_response(self, resp_text):
        """生の応答をパースして標準化された形式に変換"""
        detections = json.loads(resp_text)
        parsed = []
        for obj in detections:
            parsed.append({
                "label": obj.get("label", "object"),
                "box_2d": [
                    obj["y_min"], obj["x_min"],
                    obj["y_max"], obj["x_max"]
                ]
            })
        return parsed

    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """YOLO形式のデータセットを作成"""
        os.makedirs(output_folder_path, exist_ok=True)

        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt is None:
            prompt = (f"Detect all prominent items from the following list: {class_names} in the image. "
                      "The response should be a JSON array. Each object should have a 'label' and 'box_2d'. "
                      "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.")

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')
                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")
class GeminiInference:
    """
    Gemini API 呼び出しを扱うクラス。
    """
    def __init__(self, api_key_source=None):
        self.api_key_source = api_key_source
        if api_key_source is None and IS_COLAB:
            try:
                from google.colab import userdata
                self.api_key_source = userdata.get('gemini')
            except (ImportError, NameError):
                print("警告: Colab環境で 'gemini' のAPIキーが取得できませんでした。")

    def get_response(self, file_path, prompt):
        """
        画像ファイルに対して Geminin API 呼び出しを行い、レスポンステキストを返す。
        """
        client = genai.Client(api_key=self.api_key_source)
        my_file = client.files.upload(file=file_path)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[my_file, prompt],
        )
        return response.text

    def parse_response(self, text):
        """
        レスポンス JSON をパース。'label' と 'box_2d'([0-1000]正規化) を取り出し、[0,1]正規化に変換して返すリスト。
        """
        print(text)
        json_str = text
        if '```json' in text:
            json_str = text[text.find('```json') + len('```json'):]
        json_str = json_str.strip('` \n')
        try:
            data = json.loads(json_str)
        except Exception as e:
            print("JSON パースエラー:", e)
            return []
        if isinstance(data, dict):
            data = [data]
        parsed = []
        for obj in data:
            if 'box_2d' in obj and 'label' in obj:
                coords = obj['box_2d']
                norm = [c / 1000.0 for c in coords]
                parsed.append({'label': obj['label'], 'box_2d': norm})
        return parsed
    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        指定されたフォルダ内の全画像に対して推論を行い、YOLO形式の学習データを作成する。
        内部で自身の get_response と parse_response を呼び出す。
        """
        os.makedirs(output_folder_path, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt == None:
            prompt = (f"Detect all prominent items from the following list: {class_names} in the image. "
                      "The response should be a JSON array. Each object should have a 'label' and 'box_2d'. "
                      "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.")
           

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # 自身の推論メソッドを呼び出す
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')

                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")

class COCOEvaluator:
    """
    COCO データセットに対する Gemini モデル評価を行うクラス。
    """
    def __init__(self, coco_handler, inference, prompts, save_dir):
        self.coco = coco_handler
        self.inference = inference
        self.prompts = prompts
        self.save_dir = save_dir

    def evaluate_category(
        self,
        category_name,
        num_images=10,
        iou_threshold=0.5,
        save_images=True,
        fixed_image_ids=None
    ):
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        # 1️⃣ 評価対象画像 ID を決定
        if fixed_image_ids is not None:
            img_ids = fixed_image_ids
        else:
            all_ids = self.coco.get_image_ids(category_name)
            img_ids = random.sample(all_ids, min(num_images, len(all_ids)))

        if len(img_ids) == 0:
            print(f"カテゴリ {category_name} の画像が見つかりません。")
            return {}
        sampled_ids = random.sample(img_ids, min(num_images, len(img_ids)))

        total_iou = 0.0
        total_objs = 0
        correct_count = 0

        for img_id in sampled_ids:
            img_info = self.coco.coco.loadImgs(img_id)[0]
            img_path = self.coco.download_image(img_info)
            # GT ボックス取得
            ann_ids = self.coco.coco.getAnnIds(
                imgIds=img_id,
                catIds=self.coco.coco.getCatIds(catNms=[category_name]),
                iscrowd=False
            )
            anns = self.coco.coco.loadAnns(ann_ids)
            gt_boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                ymin = y / img_info['height']
                xmin = x / img_info['width']
                ymax = (y + h) / img_info['height']
                xmax = (x + w) / img_info['width']
                gt_boxes.append([ymin, xmin, ymax, xmax])
            if not gt_boxes:
                continue

            # プロンプト取得（辞書に無ければデフォルト文）
            prompt = self.prompts.get(
                category_name,
                f"Detect all prominent '{category_name}' items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
            )
            resp_text = self.inference.get_response(img_path, prompt)
            preds = self.inference.parse_response(resp_text)
            # ラベル一致する予測だけ抽出（小文字比較）

            #preds_cat = [p for p in preds if (p['label'].lower() in category_name.lower())== True or p['label'].lower() =='object']
            preds_cat = preds
            print(preds_cat)
            pred_boxes = [p['box_2d'] for p in preds_cat]

            # 各 GT ボックスに対し、IoU 最大の予測を探す
            for gt in gt_boxes:
                best_iou = 0.0
                best_pred = None
                for pb in pred_boxes:
                    iou = compute_iou(gt, pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pb
                total_iou += best_iou
                total_objs += 1
                if best_iou >= iou_threshold:
                    correct_count += 1

                # IoU < threshold の場合にアノテーション画像保存
                if save_images and best_iou < iou_threshold:
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    # GT を赤で描画
                    ymin, xmin, ymax, xmax = gt
                    x1, y1 = int(xmin * w), int(ymin * h)
                    x2, y2 = int(xmax * w), int(ymax * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 予測を青で描画（あれば）
                    if best_pred:
                        pymin, pxmin, pymax, pxmax = best_pred
                        px1, py1 = int(pxmin * w), int(pymin * h)
                        px2, py2 = int(pxmax * w), int(pymax * h)
                        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    save_path = os.path.join(self.save_dir, f"{category_name}_{img_id}_anno.jpg")
                    cv2.imwrite(save_path, img)

        avg_iou = total_iou / total_objs if total_objs > 0 else 0.0
        accuracy = correct_count / total_objs if total_objs > 0 else 0.0
        summary = {
            'category': category_name,
            'num_images_evaluated': len(sampled_ids),
            'total_objects': total_objs,
            'average_iou': avg_iou,
            f'accuracy@iou{iou_threshold}': accuracy
        }
        return summary

    def evaluate_all(self, num_per_category=100, iou_threshold=0.1, save_images=True):
        """
        prompts にある全カテゴリをループして評価を実行。
        結果一覧を JSON ファイルにも保存する。
        """
        results = []
        for cat in self.prompts.keys():
            print(f"Evaluating category: {cat}")
            res = self.evaluate_category(
                category_name=cat,
                num_images=num_per_category,
                iou_threshold=iou_threshold,
                save_images=save_images
            )
            results.append(res)
        # サマリ JSON 保存
        summary_path = os.path.join(self.save_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary to {summary_path}")
        return results

def download_file(url: str, dst_path: str, chunk_size: int = 1024):
    """
    URL からファイルをストリーミングダウンロードして dst_path に保存する。
    既に存在する場合はダウンロードをスキップ。
    """
    if os.path.exists(dst_path):
        print(f"ファイルが既に存在します: {dst_path}。ダウンロードをスキップします。")
        return
    # ストリーミングダウンロード
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(dst_path)}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("ダウンロード完了:", dst_path)

def extract_zip(zip_path: str, extract_dir: str, target_members: list = None):
    """
    zip_path の ZIP を展開。target_members にリストを渡すと、ZIP 内のその相対パスのみを抽出する。
    例: target_members = ['annotations/instances_val2017.json']
    None の場合はすべて解凍する。
    """
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        if target_members is None:
            z.extractall(path=extract_dir)
            print("すべて解凍しました。")
        else:
            # メンバーが ZIP 内に存在するかチェック
            zip_names = z.namelist()
            for member in target_members:
                if member in zip_names:
                    print(f"  Extracting {member} ...")
                    z.extract(member, path=extract_dir)
                else:
                    print(f"  警告: {member} が ZIP 内に見つかりません。")
    print("解凍処理完了。")

def prepare_coco_annotations(
    data_dir: str = 'coco_data',
    annotation_zip_url: str = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    extract_train: bool = False,
    extract_val: bool = True
) -> dict:
    """
    COCO アノテーション JSON を data_dir 以下に準備する。

    :param data_dir: ダウンロード・解凍先ディレクトリ
    :param annotation_zip_url: COCO のアノテーション ZIP URL
    :param extract_train: True の場合 'instances_train2017.json' も抽出
    :param extract_val: True の場合 'instances_val2017.json' を抽出
    :return: {
        'zip_path': <ZIP ファイルのローカルパス>,
        'train_json': <instances_train2017.json のパス または None>,
        'val_json': <instances_val2017.json のパス または None>
    }
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_fname = os.path.basename(annotation_zip_url)
    zip_path = os.path.join(data_dir, zip_fname)
    # 1) ダウンロード
    download_file(annotation_zip_url, zip_path)
    # 2) 解凍
    # ZIP 内で、JSON は 'annotations/instances_train2017.json' および 'annotations/instances_val2017.json' に含まれる
    target_members = []
    if extract_train:
        target_members.append('annotations/instances_train2017.json')
    if extract_val:
        target_members.append('annotations/instances_val2017.json')
    # None ならすべて解凍だが、大きいので必要なものだけを抽出するのが望ましい
    if target_members:
        extract_zip(zip_path, data_dir, target_members=target_members)
    else:
        # もし両方 False なら全部解凍（非推奨）
        extract_zip(zip_path, data_dir, target_members=None)
    # 3) 抽出後のファイルパスを返す
    train_json_path = None
    val_json_path = None
    if extract_train:
        train_json_path = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    if extract_val:
        val_json_path = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    return {
        'zip_path': zip_path,
        'train_json': train_json_path if os.path.exists(train_json_path or '') else None,
        'val_json': val_json_path if os.path.exists(val_json_path or '') else None
    }


# New Class: EnsembleInference
class EnsembleInference:
    """
    複数の推論モデルの結果を統合し、重なる領域のオブジェクト検出を強化するクラス。
    """
    def __init__(self, inference_models: dict):
        """
        :param inference_models: モデル名とInferenceクラスのインスタンスをマッピングする辞書
                                 例: {'gemini': GeminiInference(), 'kosmos': KosmosInference()}
        """
        self.inference_models = inference_models
        # available_models を順序付きのリストとして保持し、辞書にもアクセスできるようにする
        self._available_models_list = list(inference_models.keys())
        print(f"利用可能なモデル: {', '.join(self._available_models_list)}")

    def get_available_models_dict(self) -> dict:
        """
        利用可能なAIモデルのリストと、それに対応するインデックス（1-based）を辞書で返す。
        """
        model_info = {}
        print("利用可能なAIモデル:")
        for i, model_name in enumerate(self._available_models_list):
            model_info[str(i+1)] = model_name # 1-based index as string key
            model_info[model_name] = self.inference_models[model_name] # Model instance for direct access
            print(f"{i+1}: {model_name}")
        return model_info

    def select_models(self, selection_criteria=None):
        """
        協力させたいAIモデルを選択する。
        - None: 全てのモデルを使用
        - list of int/str: 指定されたインデックス（1-based）またはモデル名のみを使用
        - dict with 'exclude_indices' or 'exclude_names': 指定されたモデルを除外して使用
        :param selection_criteria: モデルの選択基準
        :return: 選択されたモデルのインスタンスのリスト
        """
        current_models = list(self.inference_models.items()) # (name, instance) pairs

        if selection_criteria is None:
            return [instance for name, instance in current_models]

        if isinstance(selection_criteria, dict):
            exclude_indices = selection_criteria.get('exclude_indices', [])
            exclude_names = selection_criteria.get('exclude_names', [])

            # Convert exclude_indices to names
            excluded_by_index = set()
            for idx in exclude_indices:
                if isinstance(idx, int) and 0 < idx <= len(self._available_models_list):
                    excluded_by_index.add(self._available_models_list[idx - 1])
                else:
                    print(f"警告: 無効な除外インデックス {idx} です。スキップします。")

            # Combine all excluded names
            all_excluded_names = excluded_by_index.union(set(exclude_names))

            selected_models = []
            for name, instance in current_models:
                if name not in all_excluded_names:
                    selected_models.append(instance)

            if not selected_models:
                print("警告: すべてのモデルが除外されたか、有効なモデルが選択されませんでした。全てのモデルを使用します。")
                return [instance for name, instance in current_models]
            return selected_models

        elif isinstance(selection_criteria, list):
            selected_models = []
            available_model_map = {name: instance for name, instance in current_models}

            for item in selection_criteria:
                if isinstance(item, int):
                    if 0 < item <= len(self._available_models_list):
                        model_name = self._available_models_list[item - 1]
                        selected_models.append(available_model_map[model_name])
                    else:
                        print(f"警告: 無効なモデルインデックス {item} です。スキップします。")
                elif isinstance(item, str):
                    if item in available_model_map:
                        selected_models.append(available_model_map[item])
                    else:
                        print(f"警告: 認識できないモデル名 '{item}' です。スキップします。")
                else:
                    print(f"警告: 無効な選択形式 '{item}' です。スキップします。")

            if not selected_models:
                print("警告: 有効なモデルが選択されませんでした。全てのモデルを使用します。")
                return [instance for name, instance in current_models]
            return selected_models
        else:
            print("警告: 無効な selection_criteria 形式です。全てのモデルを使用します。")
            return [instance for name, instance in current_models]


    def get_response(self, image_path: str, prompt: str, iou_threshold: float = 0.5, selection_criteria=None):
        """
        指定されたAIモデル群で推論を実行し、IoUが閾値以上の重なり合う領域を統合して返す。
        :param image_path: 推論対象の画像パス
        :param prompt: 各モデルに与えるプロンプト
        :param iou_threshold: 統合するバウンディングボックスのIoU閾値
        :param selection_criteria: モデルの選択基準。詳細は select_models を参照。
        :return: 統合された検出結果のリスト (Geminiと同じ形式: [{'label': '...', 'box_2d': [ymin, xmin, ymax, xmax]}, ...])
        """

        selected_inference_instances = self.select_models(selection_criteria)

        all_detections = []
        for model_instance in selected_inference_instances:
            # モデルインスタンスから名前を逆引き
            model_name = next((name for name, instance in self.inference_models.items() if instance == model_instance), "Unknown Model")
            print(f"--- Running inference with {model_name} ---")
            try:
                raw_response = model_instance.get_response(image_path, prompt)
                parsed_detections = model_instance.parse_response(raw_response)
                # モデル名を検出結果に追加して、どのモデルからのものか追跡できるようにする
                for det in parsed_detections:
                    det['source_model'] = model_name
                all_detections.extend(parsed_detections)
            except Exception as e:
                print(f"エラー: {model_name} での推論中に問題が発生しました: {e}")
                continue

        if not all_detections:
            print("すべてのモデルで検出結果がありませんでした。")
            return []

        # 重複する検出結果を統合する (NMSに似たロジック)

        # 同じラベルの検出をグループ化
        grouped_detections = {}
        for det in all_detections:
            label = det['label'].lower() # ラベルは大文字小文字を区別しない
            if label not in grouped_detections:
                grouped_detections[label] = []
            grouped_detections[label].append(det)

        integrated_results = []
        for label, detections_with_same_label in grouped_detections.items():
            boxes_to_process = [(det['box_2d'], False) for det in detections_with_same_label]

            while any(not processed for _, processed in boxes_to_process):
                # 未処理の最初のボックスを選択
                current_box_idx = -1
                for i, (box_a, processed_a) in enumerate(boxes_to_process):
                    if not processed_a:
                        current_box_idx = i
                        break

                if current_box_idx == -1: # すべて処理済み
                    break

                current_box = list(boxes_to_process[current_box_idx][0]) # リストとしてコピー
                boxes_to_process[current_box_idx] = (boxes_to_process[current_box_idx][0], True) # 処理済みとしてマーク

                # 他の未処理ボックスと比較し、IoUが閾値以上なら統合
                for j in range(len(boxes_to_process)):
                    box_b, processed_b = boxes_to_process[j]
                    if not processed_b and current_box_idx != j:
                        iou = compute_iou(current_box, box_b)
                        if iou >= iou_threshold:
                            # 重なり合う部分を計算して新しいボックスとする
                            inter_ymin = max(current_box[0], box_b[0])
                            inter_xmin = max(current_box[1], box_b[1])
                            inter_ymax = min(current_box[2], box_b[2])
                            inter_xmax = min(current_box[3], box_b[3])

                            if (inter_xmax - inter_xmin) > 0 and (inter_ymax - inter_ymin) > 0:
                                current_box = [inter_ymin, inter_xmin, inter_ymax, inter_xmax]
                                boxes_to_process[j] = (box_b, True) # 処理済みとしてマーク

                # 統合されたボックスを追加
                integrated_results.append({
                    'label': label,
                    'box_2d': current_box
                })

        print("--- Ensemble Inference Results ---")
        print(integrated_results)
        return integrated_results

    def parse_response(self, resp_data):
        """
        get_response がすでにパース済みデータを返すため、そのまま返す。
        COCOEvaluator がこのメソッドを呼び出すことを想定。
        """
        return resp_data

    def create_yolo_dataset(self, image_folder_path, output_folder_path, class_mapping, prompt=None):
        """
        指定されたフォルダ内の全画像に対して推論を行い、YOLO形式の学習データを作成する。
        内部で自身の get_response と parse_response を呼び出す。
        """
        os.makedirs(output_folder_path, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"警告: '{image_folder_path}' に画像ファイルが見つかりません。")
            return

        class_names = ", ".join(f"'{name}'" for name in class_mapping.keys())
        if prompt == None:
            prompt = (f"Detect all prominent items from the following list: {class_names} in the image. "
                      "The response should be a JSON array. Each object should have a 'label' and 'box_2d'. "
                      "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.")
           

        print(f"--- YOLOデータセット作成開始 ({self.__class__.__name__}) ---")
        print(f"使用するプロンプト: {prompt}")

        for filename in tqdm(image_files, desc=f"Processing images with {self.__class__.__name__}"):
            image_path = os.path.join(image_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder_path, f"{base_filename}.txt")

            try:
                # 自身の推論メソッドを呼び出す
                raw_response = self.get_response(image_path, prompt)
                detections = self.parse_response(raw_response)

                yolo_lines = []
                for det in detections:
                    label = det.get('label', '').lower()
                    
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        box_2d = det.get('box_2d')

                        if box_2d and len(box_2d) == 4:
                            x_center, y_center, w, h = _convert_to_yolo_format(box_2d)
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

        print(f"--- 処理完了 ---")
        print(f"YOLO形式のアノテーションファイルが '{output_folder_path}' に保存されました。")


# ローカル環境用の追加設定
if not IS_COLAB:
    # 必要なライブラリのインポートチェック
    try:
        import moondream as md
    except ImportError:
        print("moondreamがインストールされていません。必要に応じてインストールしてください")
    
    # その他のローカル環境固有の設定
    print("ローカル環境で実行中です")