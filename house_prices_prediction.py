import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# ----- 1. データ読み込みと結合 -----

# ファイルパス
train_path = 'data/train.csv'
test_path = 'data/test.csv'
submission_output_path = 'data/submit/Ver.1.0/london_prices_submission.csv' 

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

# 不要なID列の削除
df_combined.drop('ID', axis=1, inplace=True) 


# ----- 2. 特徴量エンジニアリングと前処理 -----

# 2-1. 地理情報からの特徴量抽出

# 'postcode'からOutcode（郵便番号の最初の地域コード）を抽出
# 例: 'SW1A 0AA' -> 'SW1A'
df_combined['Outcode'] = df_combined['postcode'].apply(lambda x: str(x).split(' ')[0])

# 2-2. 欠損値処理
# 欠損値が多く、欠損に意味がある特定のカテゴリ変数を'None'として処理
for col in ['tenure', 'propertyType', 'currentEnergyRating']:
    if col in df_combined.columns:
        df_combined[col].fillna('None', inplace=True)

# 緯度/経度 (latitude/longitude): 欠損値は中央値で補完
df_combined['latitude'].fillna(df_combined['latitude'].median(), inplace=True)
df_combined['longitude'].fillna(df_combined['longitude'].median(), inplace=True)

# 2-3. カテゴリ変数の処理

# 順序性のあるカテゴリ変数: Label Encoding
cols_label_encode = ['currentEnergyRating']
for col in cols_label_encode:
    if col in df_combined.columns:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col].astype(str))

# 順序性のないカテゴリ変数 ('Outcode'を含む) をOne-Hot Encoding
cols_to_dummy = ['propertyType', 'tenure', 'Outcode']
df_combined = pd.get_dummies(df_combined, columns=cols_to_dummy, drop_first=True)

# 'postcode', 'fullAddress', 'outcode', 'country' は既に処理済みまたは不要なので削除
df_combined.drop(['postcode', 'fullAddress', 'outcode', 'country'], axis=1, inplace=True, errors='ignore')

# 2-4. 最終的な欠損値チェック (残った数値欠損値は0で埋める)
df_combined.fillna(0, inplace=True)
print("特徴量エンジニアリング完了。")

# ----- 3. データセットの分離とモデル訓練 -----

# データの分離
X_train = df_combined.iloc[:len(df_train)]
X_test = df_combined.iloc[len(df_train):]

# モデルの定義 (回帰問題に強いXGBoost Regressor)
xgb_model = XGBRegressor(n_estimators=300, 
                         learning_rate=0.05, 
                         max_depth=4, 
                         random_state=42, 
                         objective='reg:squarederror') # 目的関数は二乗誤差

# 訓練の実行
xgb_model.fit(X_train, target)
print("モデル訓練完了。")

# ----- 4. 予測と提出ファイルの作成 -----

# 予測の実行 (対数変換された値が返される)
predictions_log = xgb_model.predict(X_test)

# 予測結果を元のスケールに戻す 
predictions_original = np.expm1(predictions_log)

# 提出ファイル形式に整形
submission = pd.DataFrame({'ID': df_test['ID'], 'price': predictions_original}) 

# 提出ファイルを出力
submission.to_csv(submission_output_path, index=False)

print(f"\n提出ファイル '{submission_output_path}' が正常に生成されました。")
print(f"予測価格の最初の5行:\n{submission.head()}")