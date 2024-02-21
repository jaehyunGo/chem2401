#현재 경로 확인
import os
print(os.getcwd())

#경로 변경
os.chdir('/content/sample_data/')

#파일불러오기(na_values 설정)
#공백이나 Missing표기 등을 모두 NaN이라는 표시로 변경
import pandas as pd
df = pd.read_excel('tox21_ac50.xlsx', na_values=['','  ', ' ','NA', 'N/A', 'Missing'])
print(df)

# 데이터 프레임 출력
print()
print('='*100)
print('데이터 프레임 출력')
print(df)

# 데이터프레임 칼럼명 출력
print()
print('='*100)
print('데이터프레임 칼럼명 출력')
print(df.columns)

# 데이터프레임 정보 출력(칼럼명, null값, Dtype확인 가능)
print()
print('='*100)
print('데이터프레임 정보 출력(칼럼명, null값, Dtype확인 가능)')
print(df.info())

# 데이터 프레임 처음부터 5개의 행 출력
print()
print('='*100)
print('데이터 프레임 처음부터 5개의 행 출력')
print(df.head())

# 데이터 프레임 끝에서부터 10개의 행 출력
print()
print('='*100)
print('데이터 프레임 끝에서부터 10개의 행 출력')
print(df.tail(10))

# null값 개수 출력
print()
print('='*100)
print('null값 개수 출력')
missing_values_count = df.isna().sum()
print(missing_values_count)

# SMILES null값 제거
#SMILES null값 제거
df = df.dropna(subset=['SMILES'])

# null값 행 출력
print()
print('='*100)
print('null값 행 출력')
print(df[df.isnull().any(1)])

# Null값이 존재하는지 재확인
print()
print('='*100)
print('Null값이 존재하는지 재확인')
print(df.info())

# 데이터 통계값 확인
print('평균:', df['AC50'].mean())  #평균
print('분산:', df['AC50'].var())  #분산
print('표준편차:', df['AC50'].std())  #표준편차
print('최소값:', df['AC50'].min())  #최소값
print('최대값:', df['AC50'].max())  #최대값
print('중앙값:', df['AC50'].median())  #중앙값
print('분위수:', df['AC50'].quantile(0.25))  #분위수
print('4분위수:', df['AC50'].quantile(q=[0.25, 0.5, 0.75]))  #4분위수
print('전체 통계값:', df['AC50'].describe())  #전체 통계값

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()

df['AC50'].plot(title='default-line', figsize=(20,3))
plt.show()

df['AC50'].plot(title='histogram',
                     kind='hist',
                     figsize=(3,3))
plt.show()

df['AC50'].plot(title='boxplot',
                     kind='box',
                     figsize=(3,3))
plt.show()

plt.subplot(1,2,1)
plt.plot(df['AC50'],'o', ms=3)
plt.subplot(1,2,2)
plt.plot(df['AC50'],'^',color='red',ms=3)

plt.tight_layout()
plt.show()

# stemgraphic 라이브러리 설치하기(!pip install stemgraphic)

import stemgraphic

stemgraphic.stem_graphic(df.AC50)

df.AC50.plot(kind='box',title='Boxplot',ylabel='AC50',figsize=(3,3))
plt.xticks([])
plt.show()

plt.figure(figsize=(3,3))
plt.boxplot(df.AC50)
plt.ylabel('AC50')
plt.title('BoxPlot')
plt.xticks([])
plt.show()

plt.figure(figsize=(3,3))
sns.boxplot(df.AC50)
plt.ylabel('AC50')
plt.title('BoxPlot')
plt.xticks([])
plt.show()

plt.subplot(1,2,1)
sns.histplot(data=df, x='AC50')
plt.title('(1)')

plt.subplot(1,2,2)
sns.histplot(data=df, x='AC50', bins=50)
plt.title('(2)')


plt.tight_layout()
plt.show()