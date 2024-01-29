import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(label_idx, file, train_size= 0.8, stratify= True):
    '''
        label_idx (int) : 0 ~ 11중 선택  | 0 -> NR-AR, 11 -> SR-p53
        file (string) : preprocessing한 데이터 파일의 path
        train_size (float) : train과 test의 비율
        stratify (Bool) : True - test 데이터셋과 train 데이터셋의 label의 분포가 유사하도록 split
    '''

    # 데이터 불러오기
    df = pd.read_csv(file)
    feature = df.iloc[:,1:168]
    label = df.iloc[:,168+label_idx]

    # label 기준으로 None이 들어있는 데이터셋 날리기
    idx_none = df[df.iloc[:,168 + label_idx].notna()].index
    feature = df.iloc[idx_none, 1:168]
    label = df.iloc[idx_none, 168+label_idx]

    # print('Total feature - shape:',feature.shape)
    # print('Total label - shape:',label.shape)
    
    # 데이터셋 나누기
    if stratify:
        train_x, test_x, train_y, test_y = train_test_split(feature, label, train_size= train_size, stratify= label)
    else:
        train_x, test_x, train_y, test_y = train_test_split(feature, label, train_size= train_size)

    return train_x, test_x, train_y, test_y

if __name__ == '__main__':
    label_idx = 1   # 0~11중 선택 0 -> NR-AR, 11 -> SR-p53
    folder = './Data'
    file = 'tox21_dataset.csv'
    path = f'{folder}/{file}'

    train_x, test_x, train_y, test_y = split_dataset(0,path)

    print('train dataset')
    print(f'\tX : {train_x.shape}, Y : {train_y.shape}')
    print('test dataset')
    print(f'\tX : {test_x.shape}, Y : {test_y.shape}')