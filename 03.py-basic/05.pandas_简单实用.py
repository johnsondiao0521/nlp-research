import pandas as pd
import os


if __name__ == '__main__':
    # df = pd.read_csv("data\m_result.csv")
    # print("")

    all_data = pd.read_excel(os.path.join('data', 'info.xls'))

    all_name = all_data["姓名"]
    all_num = all_data["学号"]
    all_age = all_data["年龄"]

    # all_name = list(all_data["姓名"])
    # all_num = all_data["学号"].tolist()
    # all_age = all_data["年龄"].to_list()

    all_data = dict(all_data)
    print(all_data)

    dataf = pd.DataFrame({"姓名": all_name, "年龄": all_age, "学号": all_num})
    dataf.to_excel(os.path.join('data', 'new_info.xlsx'))
    dataf.to_csv(os.path.join('data', 'new_info.csv'))
