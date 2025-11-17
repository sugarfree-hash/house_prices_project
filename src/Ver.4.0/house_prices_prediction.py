import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# ---------------------------------
# ----- 1. データ読み込みと結合 -----
# ---------------------------------

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

# ------------------------------------------
# ----- 2. 特徴量エンジニアリングと前処理 -----
# ------------------------------------------

# 2-1. 処理用の定数・リストをセット
# ロンドン中心地（トラファルガー広場付近）の座標を使用
CENTER_LAT = 51.5074
CENTER_LON = 0.1278

# 高度な補完が必要なものをリスト化
cols_for_imputation = ['bedrooms', 'bathrooms', 'livingRooms', 'floorAreaSqM']

# 2-2. 欠損値処理

# 値のないもの
for col in ['tenure', 'propertyType', 'currentEnergyRating']:
    if col in df_combined.columns:
        df_combined[col].fillna('None', inplace=True)

# 緯度・経度の補完
for col in ['latitude', 'longitude']:
    if col in df_combined.columns:
        # 欠損値をその列の中央値で補完
        df_combined[col].fillna(df_combined[col].median(), inplace=True)

# A. 欠損値フラグ（欠損していること自体が重要となりうるもの）
for col in cols_for_imputation:
    # 欠損しているなら1,そうでないなら0の列を作成
    df_combined[f'{col}_Is_Missing'] = df_combined[col].isnull().astype(int)

# B. グループ補完
# 地域の平均的な部屋数/面積で補完

# 'postcode'からOutcode（地域コード）を抽出
if 'Outcode' not in df_combined.columns:
    df_combined['Outcode'] = df_combined['postcode'].apply(lambda x: str(x).split(' ')[0])

for col in cols_for_imputation:
    if col in df_combined.columns:
        # Outcode（地域）ごとの中央値で補完
        # .transform()で元の欠損箇所に適用
        df_combined[col] = df_combined.groupby('Outcode')[col].transform(
            lambda x: x.fillna(x.median())
        )

# C. Bでグループ補完できなかった値の処理
for col in cols_for_imputation:
    if df_combined[col].isnull().any():
        df_combined[col].fillna(df_combined[col].median(), inplace=True)

# 2-3. ロンドン中心地からの距離計算
df_combined['distance_to_center'] = np.sqrt(
    (df_combined['latitude'] - CENTER_LAT)**2 + 
    (df_combined['longitude'] - CENTER_LON)**2
)

# 2-4. カテゴリ変数の処理
cols_label_encode = ['currentEnergyRating']
for col in cols_label_encode:
    if col in df_combined.columns:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col].astype(str))

cols_to_dummy = ['propertyType', 'tenure', 'currentEnergyRating', 'Outcode']
df_combined = pd.get_dummies(df_combined, columns=cols_to_dummy, dummy_na=False)

# 'postcode', 'fullAddress', 'outcode', 'country' は削除
df_combined.drop(['postcode', 'fullAddress', 'outcode', 'country'], axis=1, inplace=True, errors='ignore')

# 2-5. 最終的な欠損値チェック
df_combined.fillna(0, inplace=True)
print("特徴量エンジニアリング完了。")

# ---------------------------------------------------------
# ----- 3. データセットの分離とモデル訓練（チューニング） -----
# ---------------------------------------------------------

# データの分離
X_train = df_combined.iloc[:len(df_train)]
X_test = df_combined.iloc[len(df_train):]


# 3-1. ハイパーパラメータの探索範囲を定義（正規化を強化し過学習防止）
param_distributions = {
    'n_estimators': [300, 500, 700],     # 決定木の数
    'max_depth': [3, 5, 7],              # 決定木の深さ
    'learning_rate': [0.01, 0.05, 0.1],  # 学習率
    'gamma' :[0, 0.1, 0.5, 1],           # 決定木の分割制御
    'reg_alpha':[0, 0.001, 0.01, 0.1],   # L1正規化
}

# 3-2. モデルのインスタンス化
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

# 3-3. K-Fold交差検証の設定
# 5分割交差検証
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3-4. RandomizedSearchCVの設定
rscv = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=80, # 試行回数
    scoring='neg_mean_squared_error',
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1  # 全てのCPUコアを使用して並列処理を実行
)

# 3-5. チューニングの実行
print("ハイパーパラメータチューニングを開始します ...")
rscv.fit(X_train, target)
print("チューニング完了。")


# 3-6. 結果の表示
print("\n--- チューニング結果 ---")
print("最適パラメータ:", rscv.best_params_)
# best_score_は負の平均二乗誤差なので、平方根と負号でRMSEに戻す
best_rmse = np.sqrt(-rscv.best_score_)
print(f"ベストスコア (RMSE - 訓練データ交差検証): {best_rmse:.6f}")
print("------------------------\n")

# -------------------------------------
# ----- 4. 予測と提出ファイルの作成 -----
# ---------------------------------

# 最適モデルを使用して予測を実行
# gscv.best_estimator_ が最も性能の高かったモデル
predictions_log = rscv.best_estimator_.predict(X_test)

# 予測結果を元のスケールに戻す 
predictions_original = np.expm1(predictions_log)

# 提出ファイル形式に整形
submission = pd.DataFrame({'ID': df_test['ID'], 'price': predictions_original}) 

# 提出ファイルを出力
submission.to_csv(submission_output_path, index=False)

print(f"提出ファイル '{submission_output_path}' が正常に生成されました。")
print(f"予測価格の最初の5行:\n{submission.head()}")