import os
import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 提供されたライブラリをまとめたファイル名を anotation.py と想定
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
    EnsembleInference,
    IS_COLAB
)

# APIキーの設定（環境に合わせて書き換えてください）
if IS_COLAB:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('gemini')
    DINOX_API_TOKEN = userdata.get('dinox')
else:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    DINOX_API_TOKEN = "YOUR_DINOX_API_TOKEN"

def main():
    # --- 1. 設定 ---
    CATEGORY = 'person'
    NUM_IMAGES = 30  # 1種類30枚
    IOU_THRESHOLD = 0.5
    SAVE_DIR = "evaluation_results"
    
    # 評価対象のモデルプール
    # ※ リソースに合わせてコメントアウトして調整してください
    model_pool = {
        "Gemini": GeminiInference(api_key_source=GEMINI_API_KEY),
        "Moondream3": Moondream3Inference(),
        "GroundingDINO": GroundingDINOHFInference(),
        "DINOX": DINOXInference(api_token=DINOX_API_TOKEN),
        # "Kosmos": KosmosInference(),
        # "Moondream2": MoondreamInference(),
    }

    # --- 2. データ準備 ---
    ann_paths = prepare_coco_annotations(data_dir='coco_data', extract_val=True)
    coco_handler = COCOHandler(data_dir='coco_data', annotation_path=ann_paths['val_json'])
    
    # 全モデルで共通の画像セットを使うためにIDを固定
    all_img_ids = coco_handler.get_image_ids(category_name=CATEGORY)
    fixed_image_ids = itertools.islice(all_img_ids, NUM_IMAGES)
    fixed_image_ids = list(fixed_image_ids)

    results_list = []

    # --- 3. 全組み合わせのループ ---
    # 1個（単体）から 全個数までの組み合わせを生成
    model_names = list(model_pool.keys())
    
    for r in range(1, len(model_names) + 1):
        for combo in itertools.combinations(model_names, r):
            combo_name = "+".join(combo)
            print(f"\n[Evaluating] {combo_name}")

            # インスタンスの準備
            if len(combo) == 1:
                # 単体モデル
                current_inference = model_pool[combo[0]]
            else:
                # アンサンブルモデル
                selected_models = {name: model_pool[name] for name in combo}
                current_inference = EnsembleInference(selected_models)

            # プロンプト設定
            prompts = {CATEGORY: f"Detect {CATEGORY}."}
            
            # エバリュエーター実行
            evaluator = COCOEvaluator(
                coco_handler=coco_handler,
                inference=current_inference,
                prompts=prompts,
                save_dir=os.path.join(SAVE_DIR, combo_name)
            )

            # 評価実行
            summary = evaluator.evaluate_category(
                category_name=CATEGORY,
                num_images=NUM_IMAGES,
                iou_threshold=IOU_THRESHOLD,
                save_images=False,
                fixed_image_ids=fixed_image_ids
            )
            
            summary['model_combo'] = combo_name
            summary['model_count'] = len(combo)
            results_list.append(summary)

    # --- 4. 結果の保存と可視化 ---
    df = pd.DataFrame(results_list)
    df.to_csv("benchmark_results.csv", index=False)

    # グラフ作成
    plt.figure(figsize=(12, 8))
    # 平均IoUでソート
    df_sorted = df.sort_values('average_iou', ascending=True)
    
    colors = plt.cm.viridis(df_sorted['model_count'] / df_sorted['model_count'].max())
    
    bars = plt.barh(df_sorted['model_combo'], df_sorted['average_iou'], color=colors)
    plt.xlabel('Average IoU')
    plt.title(f'Object Detection Benchmark: {CATEGORY} (N={NUM_IMAGES})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 精度の数値をバーの横に表示
    for i, bar in enumerate(bars):
        acc = df_sorted.iloc[i][f'accuracy@iou{IOU_THRESHOLD}']
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'Acc: {acc:.2f}', va='center')

    plt.tight_layout()
    plt.savefig("benchmark_chart.png")
    plt.show()

    print("\n--- 全評価完了 ---")
    print(df[['model_combo', 'average_iou', f'accuracy@iou{IOU_THRESHOLD}']])

if __name__ == "__main__":
    main()