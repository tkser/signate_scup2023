# SIGNATE Student Cup 2023

[Competition Page](https://signate.jp/competitions/1051)<br/>
[Repository](https://github.com/tkser/signate_scup2023)

~ 2023/08/24 23:59:59 (JST)

## Data
|ヘッダ名称|値例|データ型|説明|
|--|--|--|--|
|id|1|int64|中古車の識別番号|
|region|state college|str|販売地域|
|year|2013|int64|製造年|
|manufacturer|toyota|str|製造メーカー|
|condition|fair|str|状態|
|cylinders|8 cylinders|str|気筒数|
|fuel|gas|str|使用するガソリンの種類|
|odometer|172038|int64|走行距離|
|title_status|clean|str|所有権の状態|
|transmission|automatic|str|変速機|
|drive|rwd|str|駆動方式|
|size|full-size|str|大きさ|
|type|sedan|str|ボディタイプ|
|paint_color|silver|str|色|
|state|pa|str|販売州|
|price|4724|int64|販売価格(目的変数)|

## File Structure
```bash
.
├── input
│   ├── sample_submit.csv
│   ├── train.csv
│   └── test.csv
├── output
├── work
└── exp
```


## Usage
```bash
pip install signate
poetry install
poetry run start
```

### Download Data
```bash
signate download --competition-id=1051 --path=./input
```

### Submit Data
```bash
signate submit --competition-id=1051 ./output/~~
```
