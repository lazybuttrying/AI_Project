from typing import Dict
import pandas as pd
import numpy as np
from glob import glob

INPUT_SIZE = (3, 32, 8)


class Dataset():
    def __init__(self, save_option: bool = False):
        self.save_option = save_option
        csvs, csv_imexport = self.load_dataset()
        dates = csvs["domae"]["datadate"].drop_duplicates()  # 20130212
        groups = {
            key: csvs[key].groupby("datadate") for key in csvs.keys()
        }

        # 4D : all_date x 32 x 8 x 3 (n, y, x, z)
        # https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d#c6af
        self.dataset = np.zeros((len(dates),) + INPUT_SIZE)
        self.data_idx = 0  # skip first row data: it reserved at reset()

        for i, v in enumerate(dates):
            imexport = v//100
            first, second, third = self.prepare_layer(
                v, groups, csv_imexport.loc[csv_imexport["datadate"] == imexport, :])
            self.dataset[i, 0, :, :] = first
            self.dataset[i, 1, :, :] = second
            self.dataset[i, 2, :, :] = third

        # TODO : save self.dataset for test program

    def load_dataset(self) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]]:
        files: Dict = {
            key: glob(f'dataset/train/{key}_9*') for key in ["domae", "somae", "pummok", "imexport", "weather"]
        }
        csvs = {
            key: pd.read_csv(files[key][0]) for key in ["domae", "somae", "pummok"]
        }
        csvs["weather1"] = pd.read_csv(files["weather"][0])  # .iloc[:, :-1]
        csvs["weather2"] = pd.read_csv(files["weather"][1])  # .iloc[:, :-1]
        csvs["weather3"] = pd.read_csv(files["weather"][2])  # .iloc[:, :-1]
        csvs["weather1"]["timezone"] = 1
        csvs["weather2"]["timezone"] = 0
        csvs["weather3"]["timezone"] = -1

        csvs["pummok"] = csvs["pummok"].loc[:, ~
                                            csvs["pummok"].columns.str.contains('^Unnamed')]
        csv_imexport = pd.read_csv(files["imexport"][0])

        csvs["somae"]["조사단위(kg)"] = csvs["somae"]["조사단위(kg)"].replace(
            r"g", '', regex=True).astype(float)
        csvs["somae"]["조사단위(kg)"] *= 0.001
        for k in ["domae", "somae"]:
            csvs[k].drop(["농산물 부류명", "지역명", "등급명", "시장명"],
                         axis=1, inplace=True)

        for k in csvs.keys():
            csvs[k].fillna(0.0, inplace=True)
            csvs[k].replace(r' ', '0', regex=True, inplace=True)
            csvs[k].replace(r'kg|개', '', regex=True, inplace=True)
            csvs[k] = csvs[k].astype(float)

        # for c in csvs.values():
            # print(c.columns)
        return csvs, csv_imexport

    def prepare_layer(self, date: int, groups, imexport) -> tuple[np.ndarray]:

        first = np.zeros(INPUT_SIZE[1:])
        first[:8, 0] = groups["weather1"].get_group(
            date).to_numpy().reshape(-1)
        first[8:16, 1] = groups["weather2"].get_group(
            date).to_numpy().reshape(-1)
        first[16:24, 2] = groups["weather3"].get_group(
            date).to_numpy().reshape(-1)
        first[24:30, 3] = imexport.to_numpy().reshape(-1)

        # second = pd.concat([groups["domae"].get_group(date),
        #                     groups["somae"].get_group(date).drop_duplicates(subset=["시장코드"], ignore_index=True)])
        # second.reset_index(drop=True, inplace=True)

        # * GIVE UP "somae" ~~
        second = groups["domae"].get_group(date)
        third = groups["pummok"].get_group(date)

        # first = first.to_numpy()
        second = second.drop(["datadate"], axis=1).to_numpy()
        third = third.drop(["datadate"], axis=1).to_numpy()

        # print(second.shape, second)
        second = np.pad(
            second, ((0, INPUT_SIZE[1]-second.shape[0]),
                     (0, INPUT_SIZE[2]-second.shape[1])),
            'constant', constant_values=0.0)
        third = np.pad(
            third, ((0, INPUT_SIZE[1]-third.shape[0]),
                    (0, INPUT_SIZE[2]-third.shape[1])),
            'constant', constant_values=0.0)

        if self.save_option:
            pd.DataFrame(first).to_csv(f'./why/first_{date}.csv', index=True)
            pd.DataFrame(second).to_csv(f'./why/second_{date}.csv', index=True)
            pd.DataFrame(third).to_csv(f'./why/third_{date}.csv', index=True)

        return first, second, third

    def get_next_line(self):
        self.data_idx += 1
        return self.dataset[self.data_idx, :, :, :]