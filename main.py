import os
from anotation import (
    GroundingDINOHFInference,
    prepare_coco_annotations,
    COCOHandler
)
import random

if __name__ == '__main__':
    # --- 1. COCOデータセット ダウンロード設定 ---
    # COCOデータを保存するベースディレクトリ
    COCO_DATA_DIR = 'coco_data'
    # ダウンロードしたい画像のカテゴリ
    COCO_CATEGORY = 'person'
    # ダウンロードする画像の枚数
    NUM_IMAGES = 10

    # --- 2. YOLOアノテーション作成設定 ---
    # 生成されるYOLO形式のラベルファイルを保存するディレクトリ
    OUTPUT_DIR = 'yolo_labels'
    # クラス名とIDのマッピング (COCO_CATEGORYと合わせるのが一般的)
    CLASS_MAPPING = {
        'person': 0,
        # 他に検出したいクラスがあれば追加
    }
    # GroundingDINOに与えるプロンプト (検出したい物体をピリオドで区切る)
    PROMPT = "person."

    # --- 処理開始 ---

    # --- 手順A: COCOアノテーションファイルの準備 ---
    print("--- COCOアノテーションファイルの準備を開始します ---")
    # アノテーションファイル(annotations_trainval2017.zip)をダウンロード・解凍
    ann_paths = prepare_coco_annotations(data_dir=COCO_DATA_DIR, extract_val=True)
    val_ann_path = ann_paths.get('val_json')

    if not val_ann_path:
        print("エラー: COCOの検証用アノテーションファイルが見つかりませんでした。")
        exit()
    print("アノテーションファイルの準備が完了しました。")

    # --- 手順B: COCOから画像をダウンロード ---
    print(f"--- カテゴリ'{COCO_CATEGORY}'の画像を{NUM_IMAGES}枚ダウンロードします ---")
    # COCOハンドラを初期化
    coco = COCOHandler(data_dir=COCO_DATA_DIR, annotation_path=val_ann_path, dataset_type='val2017')

    # 指定カテゴリの画像IDリストを取得
    img_ids = coco.get_image_ids(category_name=COCO_CATEGORY)
    if not img_ids:
        print(f"エラー: カテゴリ '{COCO_CATEGORY}' の画像が見つかりませんでした。")
        exit()

    # 指定枚数だけランダムに画像IDをサンプリング
    selected_ids = random.sample(img_ids, min(NUM_IMAGES, len(img_ids)))

    # 画像をダウンロード
    for img_id in selected_ids:
        img_info = coco.coco.loadImgs(img_id)[0]
        coco.download_image(img_info)

    # 画像がダウンロードされたディレクトリのパスを取得
    IMAGE_DIR = coco.images_dir
    print(f"画像のダウンロードが完了しました。保存先: {IMAGE_DIR}")

    # --- 手順C: YOLOデータセットの作成 ---
    print("\n--- GroundingDINOによるYOLOデータセット作成を開始します ---")

    # GroundingDINOモデルのインスタンスを初期化
    grounding_dino = GroundingDINOHFInference()

    # ダウンロードした画像を使ってYOLOデータセットを作成
    grounding_dino.create_yolo_dataset(
        image_folder_path=IMAGE_DIR,
        output_folder_path=OUTPUT_DIR,
        class_mapping=CLASS_MAPPING,
        prompt=PROMPT
    )

    print("--- 全ての処理が完了しました ---")