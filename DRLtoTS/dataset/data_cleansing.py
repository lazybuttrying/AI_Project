# %%
from typing import Dict
import pandas as pd
import numpy as np
from glob import glob

files: Dict = {
    key: glob(f'train/{key}_9*') for key in ["domae", "somae", "pummok", "imexport", "weather"]
}

files
# %%

csvs = {
    key: pd.read_csv(files[key][0]).fillna(0.0, inplace=True) for key in ["domae", "somae", "pummok", "imexport"]
}
csvs["weather1"] = pd.read_csv(files["weather"][0])
csvs["weather2"] = pd.read_csv(files["weather"][1])
csvs["weather3"] = pd.read_csv(files["weather"][2])

csvs["weather1"]["timezone"] = 1
csvs["weather2"]["timezone"] = 0
csvs["weather3"]["timezone"] = -1

date = csvs["domae"]["datadate"].drop_duplicates()


# %%


groups = {
    key: csvs[key].groupby("datadate") for key in csvs.keys()
}

for g in groups.keys():
    try:
        print(g)
        print(
            max([(v, groups[g].get_group(v).shape)
                for v in date], key=lambda x: x[1][0])
        )
    except:
        print("skip")
        pass


'''
domae
(20130102, (10, 10))
somae
(20140510, (62, 10))
pummok
(20130407, (122, 9))
imexport
skip
weather1
(20130101, (1, 8))
weather2
(20130101, (1, 8))
weather3
(20130101, (1, 8))
'''

# %%

groups = {
    key: csvs[key].groupby("datadate") for key in csvs.keys()
}

pummok = pd.DataFrame()


# 경매건수를 기준으로 한 이유
# 경매건수는 낮지만 거래량이 많은 것은 한 기업이 독점적으로 사들인 것이기에 제외하려고
for v in date:
    g = groups["pummok"].get_group(v).fillna(0).groupby("도매시장코드")
    for k in g.groups.keys():
        pummok = pd.concat([pummok, g.get_group(
            k).nlargest(1, '경매건수')], ignore_index=True)

pummok.to_csv("data_cleansing.csv")
pummok["datadate"].value_counts()
# %%


pummok = pd.read_csv("data_cleansing.csv")
pummok["datadate"].drop_duplicates(inplace=True)


mini, maxi = float("inf"), -float("inf")
for i, v in enumerate(pummok.index):
    if pummok.iloc[i]["해당일자_전체평균가격(원)"] == 0 or pummok.iloc[i+1]["해당일자_전체평균가격(원)"] == 0:
        continue
    x = pummok.iloc[i]["해당일자_전체평균가격(원)"] - pummok.iloc[i+1]["해당일자_전체평균가격(원)"]

    mini = min(x, mini)
    maxi = max(x, maxi)
    # 1070.774845501082 1070.774845501082
    print(pummok.iloc[i]["해당일자_전체평균가격(원)"],
          pummok.iloc[i+1]["해당일자_전체평균가격(원)"])
    print("action range :", mini, maxi)
    # action range: -526.1593168178197 417.3797787689159


# %%

new_data = pd.concat([
    day_group.get_group(date[0]),
    night_group.get_group(date[0]),
    domae_group.get_group(date[0]),
    somae_group.get_group(date[0]),
], ignore_index=True)


# %%
new_data
# %%