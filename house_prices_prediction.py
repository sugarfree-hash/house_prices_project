# ロンドン住宅価格予測コンペティションのベースラインコード（ハイパーパラメータチューニング版）
# ターゲット変数は 'price' です。

import pandas as pd
import numpy as np
# 新しくGridSearchCVをインポートします
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# ----- 1. データ読み込みと結合 -----

# ファイルパス
train_path = 'data/train.csv'
test_path = 'data/test.csv'
submission_output_path = 'data/submit/london_prices_submission.csv' 

# データ読み込み
try:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
except FileNotFoundError as e:
    print(f"\nエラー: 必須のデータファイルが見つかりません。'data/'フォルダに {e.filename} を配置してください。")
    exit()

# 'price'（ターゲット変数）を対数変換
target = np.log1p(df_train['price'])
df_train.drop('price', axis=1, inplace=True) 

# 訓練データとテストデータを結合
df_combined = pd.concat([df_train, df_test], ignore_index=True)
print(f"データ結合完了。合計行数：{len(df_combined)}")

# 不要なID列の削除 (ID列名は 'ID' で統一)
df_combined.drop('ID', axis=1, inplace=True) 


# ----- 2. 特徴量エンジニアリングと前処理 (前回から変更なし) -----

# 2-1. 地理情報からの特徴量抽出
df_combined['Outcode'] = df_combined['postcode'].apply(lambda x: str(x).split(' ')[0])

# 2-2. 欠損値処理
for col in ['tenure', 'propertyType', 'currentEnergyRating']:
    if col in df_combined.columns:
        df_combined[col].fillna('None', inplace=True)

df_combined['latitude'].fillna(df_combined['latitude'].median(), inplace=True)
df_combined['longitude'].fillna(df_combined['longitude'].median(), inplace=True)

# 2-3. カテゴリ変数の処理
cols_label_encode = ['currentEnergyRating']
for col in cols_label_encode:
    if col in df_combined.columns:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col].astype(str))

cols_to_dummy = ['propertyType', 'tenure', 'Outcode']
df_combined = pd.get_dummies(df_combined, columns=cols_to_dummy, drop_first=True)

# 'postcode', 'fullAddress', 'outcode', 'country' は削除
df_combined.drop(['postcode', 'fullAddress', 'outcode', 'country'], axis=1, inplace=True, errors='ignore')

# 2-4. 最終的な欠損値チェック
df_combined.fillna(0, inplace=True)
print("特徴量エンジニアリング完了。")

# ----- 3. データセットの分離とモデル訓練（チューニング） -----

# データの分離
X_train = df_combined.iloc[:len(df_train)]
X_test = df_combined.iloc[len(df_train):]


# 3-1. ハイパーパラメータの探索グリッドを定義
param_grid = {
    'n_estimators': [100, 300, 500],   # 決定木の数
    'max_depth': [3, 5, 7],            # 決定木の深さ
    'learning_rate': [0.05, 0.1, 0.2]  # 学習率
}

# 3-2. モデルのインスタンス化
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

# 3-3. K-Fold交差検証の設定
# n_splits=3で3分割交差検証を行います
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 3-4. GridSearchCVの設定
# scoring='neg_mean_squared_error' は、RMSEを最小化するための標準的な手法です
gscv = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=kf,
    verbose=1,
    n_jobs=-1  # 全てのCPUコアを使用して並列処理を実行
)

# 3-5. チューニングの実行
print("ハイパーパラメータチューニングを開始します (時間がかかります)...")
gscv.fit(X_train, target)
print("チューニング完了。")


# 3-6. 結果の表示
print("\n--- チューニング結果 ---")
print("最適パラメータ:", gscv.best_params_)
# best_score_は負の平均二乗誤差なので、平方根と負号でRMSEに戻します
best_rmse = np.sqrt(-gscv.best_score_)
print(f"ベストスコア (RMSE - 訓練データ交差検証): {best_rmse:.6f}")
print("------------------------\n")


# ----- 4. 予測と提出ファイルの作成 -----

# 最適モデルを使用して予測を実行
# gscv.best_estimator_ が最も性能の高かったモデルです
predictions_log = gscv.best_estimator_.predict(X_test)

# 予測結果を元のスケールに戻す 
predictions_original = np.expm1(predictions_log)

# 提出ファイル形式に整形
submission = pd.DataFrame({'ID': df_test['ID'], 'price': predictions_original}) 

# 提出ファイルを出力
submission.to_csv(submission_output_path, index=False)

print(f"提出ファイル '{submission_output_path}' が正常に生成されました。")
print(f"予測価格の最初の5行:\n{submission.head()}")