import os
import time
import json
import random
import requests
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch

# 環境検出
try:
    from google.colab import userdata, drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# 既存の自作ライブラリやクラス定義（anotation.pyに相当する内容が含まれている前提）
# ここではインターフェースを維持するためにインポート形式で記載しますが、
# 実際にはこれまでに定義したクラス一式を同じファイルかパスの通った場所に配置してください。
from anotation import (
    prepare_coco_annotations,
    COCOHandler,
    COCOEvaluator,
    GeminiInference,
    KosmosInference,
    MoondreamInference,
    Moondream3Inference,
    DINOXInference,
    GroundingDINOHFInference,
    EnsembleInference
)

def run_full_benchmark():
    # --- 1. 設定とドライブのマウント ---
    CATEGORY = 'person'
    NUM_IMAGES = 30 
    IOU_THRESHOLD = 0.5
    
    if IS_COLAB:
        print("Google Colab環境を検出しました。Googleドライブをマウントします...")
        drive.mount('/content/drive')
        # ドライブ内の保存先パス
        BASE_SAVE_DIR = '/content/drive/MyDrive/DINOX_Benchmark'
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)
        print(f"結果は以下のディレクトリに保存されます: {BASE_SAVE_DIR}")
    else:
        BASE_SAVE_DIR = "full_benchmark_results"
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # APIキーの取得
    if IS_COLAB:
        GEMINI_API_KEY = userdata.get('gemini')
        DINOX_API_TOKEN = userdata.get('dinox')
    else:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY")
        DINOX_API_TOKEN = os.getenv("DINOX_API_TOKEN", "YOUR_TOKEN")

    # --- 2. 全インファレンス・モデルの初期化 ---
    print("--- 各モデルを初期化中... ---")
    
    # すべての推論クラスをリストアップ
    model_pool = {
        "Gemini-2.5-Pro": GeminiInference(api_key_source=GEMINI_API_KEY),
        "DINO-X-API": DINOXInference(api_token=DINOX_API_TOKEN),
        "Moondream3": Moondream3Inference(),
        "Moondream2": MoondreamInference(),
        "Kosmos-2": KosmosInference(),
        "GroundingDINO-HF": GroundingDINOHFInference()
    }

    # --- 3. COCOデータの準備 ---
    print("--- COCOデータの準備中... ---")
    ann_paths = prepare_coco_annotations(data_dir='coco_data', extract_val=True)
    coco_handler = COCOHandler(data_dir='coco_data', annotation_path=ann_paths['val_json'])
    
    all_img_ids = coco_handler.get_image_ids(category_name=CATEGORY)
    if len(all_img_ids) < NUM_IMAGES:
        actual_num = len(all_img_ids)
    else:
        actual_num = NUM_IMAGES
    
    fixed_image_ids = random.sample(all_img_ids, actual_num)
    print(f"評価開始: カテゴリ={CATEGORY}, 画像枚数={actual_num}")

    results_list = []

    # --- 4. 全組み合わせのループ実行 ---
    model_names = list(model_pool.keys())
    total_combinations = 2**len(model_names) - 1
    print(f"合計 {total_combinations} 通りの組み合わせ（アンサンブル含む）を評価します。")

    for r in range(1, len(model_names) + 1):
        for combo in itertools.combinations(model_names, r):
            combo_name = " + ".join(combo)
            print(f"\n[実行中] {combo_name}")

            # 推論エンジンの選択
            if len(combo) == 1:
                current_inference = model_pool[combo[0]]
            else:
                selected_instances = {name: model_pool[name] for name in combo}
                current_inference = EnsembleInference(selected_instances)

            # 保存サブディレクトリの作成
            safe_combo_name = combo_name.replace(" + ", "_").replace("-", "_")
            combo_save_dir = os.path.join(BASE_SAVE_DIR, "details", safe_combo_name)
            os.makedirs(combo_save_dir, exist_ok=True)

            prompts = {CATEGORY: f"Detect all {CATEGORY} in the image."}
            
            evaluator = COCOEvaluator(
                coco_handler=coco_handler,
                inference=current_inference,
                prompts=prompts,
                save_dir=combo_save_dir
            )

            try:
                # 評価実行（各組み合わせの個別ラベリング結果は上記ディレクトリに保存される）
                summary = evaluator.evaluate_category(
                    category_name=CATEGORY,
                    num_images=actual_num,
                    iou_threshold=IOU_THRESHOLD,
                    save_images=True, # 全組み合わせのラベリング結果を保存
                    fixed_image_ids=fixed_image_ids
                )
                
                summary['model_combo'] = combo_name
                summary['num_models'] = len(combo)
                results_list.append(summary)
            except Exception as e:
                print(f"エラー発生 ({combo_name}): {e}")

    # --- 5. 分析結果の集計・保存・グラフ化 ---
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(BASE_SAVE_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"分析サマリを保存しました: {csv_path}")

    # グラフ描画
    plt.figure(figsize=(14, 12))
    df_sorted = df.sort_values('average_iou', ascending=True)
    
    # 組み合わせのモデル数に応じて色をグラデーション
    num_m = df_sorted['num_models'].astype(float)
    colors = plt.cm.viridis(num_m / num_m.max())
    
    bars = plt.barh(df_sorted['model_combo'], df_sorted['average_iou'], color=colors)
    plt.xlabel('Average IoU')
    plt.title(f'Comprehensive Benchmark Results ({CATEGORY}, N={actual_num})\nSaved to Google Drive')
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='IoU Threshold 0.5')
    
    for i, bar in enumerate(bars):
        acc = df_sorted.iloc[i][f'accuracy@iou{IOU_THRESHOLD}']
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'Acc: {acc:.1%}', va='center', fontsize=8)

    plt.tight_layout()
    graph_path = os.path.join(BASE_SAVE_DIR, "benchmark_graph.png")
    plt.savefig(graph_path)
    plt.show()
    print(f"比較グラフを保存しました: {graph_path}")

    # 最終結果をコンソールに出力
    print("\n--- 最終ランキング (Average IoU 順) ---")
    final_view = df[['model_combo', 'average_iou', f'accuracy@iou{IOU_THRESHOLD}']].sort_values('average_iou', ascending=False)
    print(final_view)

if __name__ == '__main__':
    run_full_benchmark()