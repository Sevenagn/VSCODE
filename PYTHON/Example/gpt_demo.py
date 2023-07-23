import os
import re
import pandas as pd
import csv
from tkinter import Tk, filedialog

def write_to_excel(data_list, output_file):
    # 创建一个DataFrame对象
    # df = pd.DataFrame(data_list)

    # 将DataFrame写入Excel文件
    # df.to_excel(output_file, index=False)
    
    data_list.to_excel(output_file, index=False)

def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print("Error: The file does not exist.")
        return None

def split_records(data):
    # 定义用于分割记录的正则表达式
    record_delimiter = r"\n\n\n"

    # 分割文本数据为每条记录
    records = re.split(record_delimiter, data.strip())

    return records

def extract_transfer_records(records,input_dir):
    # 定义用于提取字段的正则表达式
    # pattern = r"转账-转给(.*?)(-?\d+\.\d+)(?:\n当前状态|¥￥P|¥P).*?\n(.*?)\n转账时间\n(.*?)\n收款时间\n(.*?)\n支付方式\n(.*?)\n转账单号\n(\d+)"
    pattern1 = r"≦(.*?)≧(.*?)转账-转给(.*?)(-?\d+\.\d+)\n当前状态\n(.*?)\n转账时间\n(.*?)\n收款时间\n(.*?)\n转账单号\n(\d+)"
    pattern2 = r"≦(.*?)≧(.*?)扫二维码付款-给(.*?)(-?\d+\.\d+)\n支付成功\n(.*?)\n转账时间\n(.*?)\n转账单号\n(\d+)"
    pattern3 = r"≦(.*?)≧(.*?)转账-转给(.*?)(-?\d+\.\d+)\n当前状态\n(.*?)\n转账时间\n(.*?)\n收款时间\n(.*?)\n转账单号\n(\d+)"
    pattern_error = r"≦(.*?)≧(.*?)"
    # pattern = r"转账-转给(.*?)\n(-?\d+\.\d+)\n当前状态\n(.*?)\n转账时间\n(.*?)\n收款时间\n(.*?)\n支付方式\n(.*?)\n转账单号\n(\d+)"
    # 存储提取出的数据
    extracted_data = []

    for record in records:
        match = re.search(pattern1, record, re.DOTALL)
        if match:
            image_name = match.group(1).strip()
            # image_path = "=HYPERLINK(\"D:\\payinfo\\2023\\"+image_name+"\")"
            image_path = "=HYPERLINK(\""+input_dir+image_name+"\")"
            temp1 = match.group(2)
            transfer_to = match.group(3).strip()
            transfer_amount = float(match.group(4).replace(',', ''))  # 去除金额中的逗号并转为浮点数
            temp2 = match.group(5).strip()  # 去除状态前后的空格
            transfer_time = match.group(6)
            temp3 = match.group(7)
            # payment_method = match.group(6)
            transfer_number = match.group(8)

            # 将提取出的数据以字典形式存储
            transfer_record = {
                "转账对象": transfer_to,
                "转账金额": transfer_amount,
                "转账时间": transfer_time,
                "转账单号": transfer_number,
                "图片名称": image_path,
                
                "识别状态": "成功",
                "未识别字段1": temp2,
                "未识别字段2": temp3

            }
            extracted_data.append(transfer_record)
        else:
            match = re.search(pattern2, record, re.DOTALL)
            if match:
                image_name = match.group(1).strip()
                # image_path = "=HYPERLINK(\"D:\\payinfo\\2023\\"+image_name+"\")"
                image_path = "=HYPERLINK(\""+input_dir+image_name+"\")"
                temp1 = match.group(2)
                transfer_to = match.group(3).strip()
                transfer_amount = float(match.group(4).replace(',', ''))  # 去除金额中的逗号并转为浮点数
                temp2 = match.group(5).strip()  # 去除状态前后的空格
                transfer_time = match.group(6)
                # temp3 = match.group(7)
                # payment_method = match.group(6)
                transfer_number = match.group(7)

                # 将提取出的数据以字典形式存储
                transfer_record = {
                    "转账对象": transfer_to,
                    "转账金额": transfer_amount,
                    "转账时间": transfer_time,
                    "转账单号": transfer_number,
                    "图片名称": image_path,
                    "识别状态": "成功",
                    "未识别字段1": temp2,
                    "未识别字段2": ""

                }
                extracted_data.append(transfer_record)
            else:
                match = re.search(pattern3, record, re.DOTALL)
                if match:
                    image_name = match.group(1).strip()
                    # image_path = "=HYPERLINK(\"D:\\payinfo\\2023\\"+image_name+"\")"
                    image_path = "=HYPERLINK(\""+input_dir+image_name+"\")"
                    temp1 = match.group(2)
                    transfer_to = match.group(3).strip()
                    transfer_amount = float(match.group(4).replace(',', ''))  # 去除金额中的逗号并转为浮点数
                    temp2 = match.group(5).strip()  # 去除状态前后的空格
                    transfer_time = match.group(6)
                    # temp3 = match.group(7)
                    # payment_method = match.group(6)
                    transfer_number = match.group(7)

                    # 将提取出的数据以字典形式存储
                    transfer_record = {
                        "转账对象": transfer_to,
                        "转账金额": transfer_amount,
                        "转账时间": transfer_time,
                        "转账单号": transfer_number,
                        "图片名称": image_path,
                        "识别状态": "成功",
                        "未识别字段1": temp2,
                        "未识别字段2": ""

                    }
                    extracted_data.append(transfer_record)
                else:
                    match = re.search(pattern_error, record, re.DOTALL)
                    if match:
                        image_name = match.group(1).strip()
                        # image_path = "=HYPERLINK(\"D:\\payinfo\\2023\\"+image_name+"\")"
                        image_path = "=HYPERLINK(\""+input_dir+image_name+"\")"
                        # 将提取出的数据以字典形式存储
                        transfer_record = {
                            "转账对象": "",
                            "转账金额": "",
                            "转账时间": "",
                            "转账单号": "",
                            "图片名称": image_path,
                            "识别状态": "失败，转账内容无法识别",
                            "未识别字段1": record,
                            "未识别字段2": ""

                        }
                        extracted_data.append(transfer_record)
                    else:
                        # 将提取出的数据以字典形式存储
                        transfer_record = {
                            "转账对象": "",
                            "转账金额": "",
                            "转账时间": "",
                            "转账单号": "",
                            "图片名称": "",
                            "识别状态": "失败，转账记录无法识别",
                            "未识别字段1": record,
                            "未识别字段2": ""

                        }
                        extracted_data.append(transfer_record)

    return extracted_data

if __name__ == "__main__":
    # 请将下面的文件路径替换为你自己的txt文件路径
    # file_path = r"D:\payinfo\2023\2023.txt"
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        input_directory = os.path.dirname(file_path)+'/'
        input_directory_new = input_directory.replace('/', '\\')

        data = read_txt_file(file_path)

        if data:
            records = split_records(data)
            transfer_records = extract_transfer_records(records,input_directory_new)
            df = pd.DataFrame(transfer_records)
            # print(df)
            df['转账对象'] = df['转账对象'].str.replace('\n', '')
            # df['转账金额'] = df['转账金额'].abs()
            df['转账金额'] = pd.to_numeric(df['转账金额'], errors='coerce').abs()
            df = df.sort_values(by='转账时间')
            # output_file = r"D:\payinfo\2023\output.xlsx"
            output_file = file_path.replace('txt', 'xlsx')
            write_to_excel(df, output_file)

            # for record in transfer_records:
            #     print(record)

