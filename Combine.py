import pandas as pd

# CSV 파일 불러오기
df1 = pd.read_csv('data/train_data.csv')
df2 = pd.read_csv('data/test_data.csv')


# answerCode가 -1인 행 제외
df_filtered = df2[df2['answerCode'] != -1]

# 두 데이터프레임 합치기
combined_df = pd.concat([df1, df_filtered])

combined_df_sorted = combined_df.sort_values(by='userID')
# 결과 저장
combined_df_sorted.to_csv('data/combined_train.csv', index=False)