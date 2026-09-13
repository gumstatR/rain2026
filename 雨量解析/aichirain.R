library(ggplot2)
library(tidyr)

# 1. データの読み込み（文字コードはUTF-8を指定）
df_2000 <- read.csv("2000raindata.csv", fileEncoding = "UTF-8")
df_2026 <- read.csv("2026raindata.csv", fileEncoding = "UTF-8")

# 2. 列名の確認
head(df_2000)
head(df_2026)

# 3. 名古屋市のデータを抽出して比較するグラフを作る例
# 日時列の名前（通常は "Unnamed.0" や "X" などになることがあります）を確認して指定してください
colnames(df_2000)[1] <- "DateTime"
colnames(df_2026)[1] <- "DateTime"

# データを結合しやすくするためにイベント列を追加
df_2000$Event <- "2000年 東海豪雨"
df_2026$Event <- "2026年 9月豪雨"

# 名古屋のデータに絞って確認
print("2000年 名古屋最大1時間雨量:")
print(max(df_2000$名古屋, na.rm = TRUE))

print("2026年 名古屋最大1時間雨量:")
print(max(df_2026$名古屋, na.rm = TRUE))

