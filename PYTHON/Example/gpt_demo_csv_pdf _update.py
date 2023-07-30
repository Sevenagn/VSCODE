import os
import time
import pandas as pd
import fitz  # PyMuPDF
from tkinter import Tk, filedialog

def highlight_trade_numbers(input_path, output_path, trade_data,input_type):
    pdf_document = fitz.open(input_path)
    matched_trade_numbers = {}  # 记录每个交易单号匹配到的次数
    ##匹配长度
    match_len = 22
    ##标注x坐标
    rect_x0 = 42
    rect_x1 = 552
    if input_type == "alipay" :
        match_len = 23
        rect_x0 = 50
        rect_x1 = 544

    for _, row in trade_data.iterrows():
        if input_type == "alipay" :
            trade_number = row["交易订单号"].strip()
        else :
            trade_number = row["交易单号"].strip()
        
        transaction_type = row["收/支"]
        truncated_trade_number_f = trade_number[:match_len]
        truncated_trade_number_s = trade_number[match_len:match_len*2]
        trade_number_found = False  # 用于标记当前交易单号是否找到匹配

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]

            # 尝试截取前match_len位进行匹配
            occurrences_f = page.search_for(truncated_trade_number_f)
            if occurrences_f:
                
                for rect_f in occurrences_f:
                    if len(truncated_trade_number_s) > 0:
                        # 尝试截取后match_len位进行匹配
                        occurrences_s = page.search_for(truncated_trade_number_s)
                        if occurrences_s:
                            for rect_s in occurrences_s:
                                if rect_s.y0 - rect_f.y0 > 0 and rect_s.y0 - rect_f.y0 < 15 and rect_s.x0 == rect_f.x0:
                                    trade_number_found = True
                                    matched_trade_numbers[trade_number] = matched_trade_numbers.get(trade_number, 0) + 1
                                    rect_f.x0 = rect_x0
                                    rect_f.x1 = rect_x1
                                    rect_f.y1 = rect_s.y1
                                    # 打印矩形的坐标信息
                                    print(f"Trade Number: {trade_number}, Page: {page_number + 1}, Rect: {rect_f}")
                                    # 添加高亮标注
                                    highlight = page.add_highlight_annot(rect_f)
                                    # 设置高亮颜色为绿色或红色
                                    highlight_color = (0, 1, 0) if transaction_type == '收入' else (1, 0, 0)
                                    highlight.set_colors(stroke=highlight_color)
                                    highlight.update()
                    else :
                        trade_number_found = True
                        matched_trade_numbers[trade_number] = matched_trade_numbers.get(trade_number, 0) + 1
                        rect_f.x0 = rect_x0
                        rect_f.x1 = rect_x1
                        # 打印矩形的坐标信息
                        print(f"Trade Number: {trade_number}, Page: {page_number + 1}, Rect: {rect_f}")
                        # 添加高亮标注
                        highlight = page.add_highlight_annot(rect_f)
                        # 设置高亮颜色为绿色或红色
                        highlight_color = (0, 1, 0) if transaction_type == '收入' else (1, 0, 0)
                        highlight.set_colors(stroke=highlight_color)
                        highlight.update()

        if not trade_number_found:
            matched_trade_numbers[trade_number] = 0

    pdf_document.save(output_path, garbage=4, deflate=True)
    pdf_document.close()

    return matched_trade_numbers

def main():
    # input_file = r"D:\wechatbills\2019.pdf"   # 输入的PDF文件路径
    root = Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(title="Select PDF file", filetypes=[("PDF files", "*.pdf")])
    if not input_file:
        print("No PDF file selected.")
        return

    input_type = "wechat"
    if "alipay" in os.path.basename(input_file):
        input_type = "alipay"

    # 构建对应的CSV文件路径
    csv_file = input_file.replace(".pdf", ".csv")
    # Check if the csv_file exists
    if not os.path.exists(csv_file):
        print(csv_file + " does not exist.")
        return

    output_file = input_file.replace(".pdf", "_marked.pdf")  # 输出的高亮后的PDF文件路径
    # 构建对应的TXT文件路径
    txt_file = input_file.replace(".pdf", "_log.txt")

    # df = pd.read_csv(csv_file)
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError as e:
        df = pd.read_csv(csv_file, encoding='gbk')

    # 获取 "收/支", "交易单号" 和 "商户单号" 列，并转换为 DataFrame
    if input_type == "alipay" :
        trade_data = df[["收/支", "交易订单号"]].dropna()
    else :
        trade_data = df[["收/支", "交易单号"]].dropna()

    total_trade_numbers = len(trade_data)

    # 记录开始时间
    start_time = time.time()

    # 根据交易单号在PDF的所有页面添加高亮，并记录匹配和未匹配的交易单号笔数
    matched_trade_numbers = highlight_trade_numbers(input_file, output_file, trade_data,input_type)

    # 记录结束时间
    end_time = time.time()

    # 计算耗时
    duration = end_time - start_time
            
    occurrences_not_equal_to_1 = sum(1 for occurrences in matched_trade_numbers.values() if occurrences != 1)
    occurrences_equal_to_1 = len(matched_trade_numbers) - occurrences_not_equal_to_1 

    with open(txt_file, "w") as f:
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        print(f"Time taken for highlighting: {duration:.2f} seconds")
        f.write(f"Time taken for highlighting: {duration:.2f} seconds\n")
        print(f"Total trade numbers in the Excel file: {total_trade_numbers}")
        f.write(f"Total trade numbers in the Excel file: {total_trade_numbers}\n")
        print(f"Legal Matched trade numbers in the PDF: {occurrences_equal_to_1}")
        f.write(f"Legal Matched trade numbers in the PDF: {occurrences_equal_to_1}\n")
        print(f"Illegal Matched trade numbers in the PDF: {occurrences_not_equal_to_1}")
        f.write(f"Illegal Matched trade numbers in the PDF: {occurrences_not_equal_to_1}\n")
        print(f"Marked PDF saved to {output_file}")
        f.write(f"Marked PDF saved to {output_file}")
        print(f"Log has been saved to {txt_file}")
        print(f"Illegal Matched trade numbers and their occurrences in the PDF:")
        f.write("Trade numbers and their occurrences in the PDF:\n")
        for trade_number, occurrences in matched_trade_numbers.items():
            f.write(f"Trade Number: {trade_number}, Occurrences: {occurrences}\n")
            if occurrences != 1:
                print(f"Trade Number: {trade_number}, Occurrences: {occurrences}")


if __name__ == "__main__":
    main()
