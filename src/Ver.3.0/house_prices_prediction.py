import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# 1. データ読み込みと結合
# ----------------------------------------------------

# ファイルパスは環境に合わせて調整してください
# 例:
# train_path = 'train.csv'
# test_path = 'test.csv'

# ダミーデータとして、以前のコードの構造を維持して読み込みをシミュレーション
# 実際にはご自身のファイルパスに置き換えてください
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Warning: CSV files not found.")

# ターゲット変数（価格）の対数変換 (正規分布に近づけ、モデル安定化)
target = np.log1p(df_train['price'])
df_train = df_train.drop('price', axis=1)

# データセットの結合（前処理の一貫性を確保するため）
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# ----------------------------------------------------
# 2. 特徴量エンジニアリングと前処理
# ----------------------------------------------------

# 2-1. 処理用の定数・リストをセット
# ロンドン中心地（トラファルガー広場付近）の座標を使用
CENTER_LAT = 51.5074
CENTER_LON = 0.1278

# 高度な補完が必要な最重要数値をリスト化
cols_for_imputation = ['bedrooms', 'bathrooms', 'livingRooms', 'floorAreaSqM']

# 2-2. 欠損値処理

# ----------------------------------------------------
# 2-2-A. シンプルな補完 (他の処理に依存しないもの)
# ----------------------------------------------------

# カテゴリ変数の欠損値処理 (文字列 'None' で補完)
for col in ['tenure', 'propertyType', 'currentEnergyRating']:
    if col in df_combined.columns:
        df_combined[col].fillna('None', inplace=True)

# 緯度・経度の補完
# 値幅が狭い連続値のため、中央値でシンプルに補完する
for col in ['latitude', 'longitude']:
    if col in df_combined.columns:
        # 欠損値をその列の中央値で補完
        df_combined[col].fillna(df_combined[col].median(), inplace=True)

# ----------------------------------------------------
# 2-2-B. 高度な連動補完 (最重要数値)
# ----------------------------------------------------

# 欠損フラグの追加 (欠損していること自体が重要となりうるもの)
for col in cols_for_imputation:
    # 欠損しているなら1,そうでないなら0の列を作成
    df_combined[f'{col}_Is_Missing'] = df_combined[col].isnull().astype(int)

# グループ補完の準備: 'postcode'からOutcode（地域コード）を抽出
if 'Outcode' not in df_combined.columns:
    # str(x).split(' ')[0] は、例 'SW1A 0AA' から 'SW1A' を抽出する処理
    df_combined['Outcode'] = df_combined['postcode'].apply(lambda x: str(x).split(' ')[0])

# グループ補完の実行: 地域の平均的な部屋数/面積で補完
for col in cols_for_imputation:
    if col in df_combined.columns:
        # Outcode（地域）ごとの中央値で補完
        # .transform()で元の欠損箇所に適用し、地域性を反映させる
        # 注意: グループ補完の処理は非常に時間がかかる場合があります
        df_combined[col] = df_combined.groupby('Outcode')[col].transform(
            lambda x: x.fillna(x.median())
        )

# Bでグループ補完できなかった値の処理
# (例: Outcodeがユニークで中央値が計算できない場合など)
for col in cols_for_imputation:
    if df_combined[col].isnull().any():
        # 全体の中央値で補完する（最終手段）
        df_combined[col].fillna(df_combined[col].median(), inplace=True)

# ----------------------------------------------------
# 2-3. 特徴量の生成
# ----------------------------------------------------

# ロンドン中心地からの距離 (Km) を再計算
df_combined['distance_to_center'] = np.sqrt(
    (df_combined['latitude'] - CENTER_LAT)**2 + 
    (df_combined['longitude'] - CENTER_LON)**2
)

# 2-4. カテゴリ変数のエンコーディング (One-Hot Encoding)

cols_to_encode = [
    'propertyType', 
    'tenure', 
    'currentEnergyRating', 
    'Outcode' # 地域コードをダミー変数化
]

# dummy_na=False: NaNは事前に'None'として処理済みのため
df_combined = pd.get_dummies(df_combined, columns=cols_to_encode, dummy_na=False)

# 2-5. 不要な列の削除と最終処理
# postcodeはOutcodeに変換済みのため不要
df_combined = df_combined.drop(['postcode'], axis=1)

# 最終的な欠損値チェックと補完（念の為、残りの欠損を0で補完）
df_combined.fillna(0, inplace=True)

print("特徴量エンジニアリング完了。")

# ----------------------------------------------------
# 3. モデル訓練と予測のためのデータ再分割
# ----------------------------------------------------

# 学習データとテストデータの再分割
X_train_processed = df_combined[:len(df_train)]
X_test_processed = df_combined[len(df_train):]

# 3-1. K-Fold交差検証の準備 (安定した評価)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3-2. モデルの初期化
xgb_model = XGBRegressor(
    random_state=42, 
    objective='reg:squarederror', 
    n_jobs=-1
)

# 3-3. ハイパーパラメータの探索範囲 
param_distributions = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}

# 3-4. Randomized Search Cross Validationの実行
# n_iter=50 (このスコア時点の設定)
rscv = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50, 
    scoring='neg_mean_squared_error',
    cv=kf,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

rscv.fit(X_train_processed, target)

# 3-5. 最適パラメータの取得
best_xgb = rscv.best_estimator_
print(f"Best Params (Score 168127): {rscv.best_params_}")

# ----------------------------------------------------
# 4. 最終予測と提出ファイルの作成
# ----------------------------------------------------

# テストデータで予測を実行
log_predictions = best_xgb.predict(X_test_processed)

# 対数変換を元に戻す (e^x - 1)
final_predictions = np.expm1(log_predictions)

# 提出ファイルの作成 (IDはテストデータに依存)
submission = pd.DataFrame({'Id': df_test.index, 'price': final_predictions})
# submission.to_csv('london_prices_submission.csv', index=False) 

print("予測ファイルの生成完了。")