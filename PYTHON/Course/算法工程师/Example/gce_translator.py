import pandas as pd
from googletrans import Translator
from tkinter import Tk, filedialog

def translate_chinese_to_thai(text):
    translator = Translator()
    translation = translator.translate(text, src='zh-cn', dest='th')
    return translation.text

def translate_and_save_to_original(input_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 翻译初始文字列并替换原始数据
    df['转换后文字'] = df['初始文字'].apply(translate_chinese_to_thai)

    # 将结果写回到原始Excel文件
    df.to_excel(input_file, index=False)

def choose_file():
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    file_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel files", "*.xlsx;*.xls")])

    return file_path

if __name__ == "__main__":
    # 用户手动选择Excel文件
    input_excel_path = choose_file()

    if not input_excel_path:
        print("未选择文件。程序退出。")
    else:
        translate_and_save_to_original(input_excel_path)
        print("翻译并保存完成。")
