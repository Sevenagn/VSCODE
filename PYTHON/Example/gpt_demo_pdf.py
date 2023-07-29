import os
import time
import pandas as pd
import fitz  # PyMuPDF

def highlight_trade_numbers(input_path, output_path, trade_numbers, highlight_color=(1, 0, 0)):
    def truncate_trade_number_length(trade_number, length):
        return trade_number[:length]

    def truncate_trade_number_index(trade_number, start_index):
        return trade_number[start_index:]

    pdf_document = fitz.open(input_path)
    matched_trade_numbers = {}  # 记录每个交易单号匹配到的次数
    not_found_trade_numbers = []   # 记录未匹配到的交易单号

    for trade_number in trade_numbers:
        truncated_trade_number_22 = trade_number[22:] + r'\b'
        truncated_trade_number_16 = trade_number[16:] + r'\b'
        trade_number_found = False  # 用于标记当前交易单号是否找到匹配

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]

            # 尝试截取前22位进行匹配
            occurrences = page.search_for(truncated_trade_number_22)
            if occurrences:
                trade_number_found = True
                matched_trade_numbers[trade_number] = matched_trade_numbers.get(trade_number, 0) + len(occurrences)
                for rect in occurrences:
                    # 添加高亮标注
                    highlight = page.add_highlight_annot(rect)
                    # 设置高亮颜色
                    highlight.set_colors(stroke=highlight_color)

        if not trade_number_found:
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]

                # 尝试截取前16位进行匹配
                occurrences = page.search_for(truncated_trade_number_16)
                if occurrences:
                    trade_number_found = True
                    matched_trade_numbers[trade_number] = matched_trade_numbers.get(trade_number, 0) + len(occurrences)
                    for rect in occurrences:
                        # 添加高亮标注
                        highlight = page.add_highlight_annot(rect)
                        # 设置高亮颜色
                        highlight.set_colors(stroke=highlight_color)

        if not trade_number_found:
            not_found_trade_numbers.append(trade_number)

    pdf_document.save(output_path, garbage=4, deflate=True)
    pdf_document.close()

    return matched_trade_numbers, not_found_trade_numbers

def main():
    input_file = r"D:\wechatbills\2019.pdf"   # 输入的PDF文件路径
    output_file = input_file.replace(".pdf", "_marked.pdf")  # 输出的高亮后的PDF文件路径
    highlight_color = (1, 0, 0)  # 高亮颜色，这里使用红色

    # 构建对应的Excel文件路径
    excel_file = input_file.replace(".pdf", ".xlsx")

    df = pd.read_excel(excel_file)

    # 获取第四列交易单号，并转换为列表
    # trade_numbers = df.iloc[:, 3].dropna().astype(str).tolist()
    # 获取"转账单号"列，并转换为列表
    trade_numbers = df["转账单号"].dropna().astype(str).tolist()

    total_trade_numbers = len(trade_numbers)

    # 记录开始时间
    start_time = time.time()

    # 根据交易单号在PDF的所有页面添加高亮，并记录匹配和未匹配的交易单号笔数
    matched_trade_numbers, not_found_trade_numbers = highlight_trade_numbers(input_file, output_file, trade_numbers, highlight_color)

    # 记录结束时间
    end_time = time.time()

    # 计算耗时
    duration = end_time - start_time

    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Time taken for highlighting: {duration:.2f} seconds")
    print(f"Total trade numbers in the Excel file: {total_trade_numbers}")
    print(f"Matched trade numbers in the PDF: {len(matched_trade_numbers)}")
    print(f"Unmatched trade numbers in the PDF: {len(not_found_trade_numbers)}")

    if len(not_found_trade_numbers) == total_trade_numbers:
        with open("not_found_trade_numbers.txt", "w") as f:
            f.write("\n".join(not_found_trade_numbers))
        print("None of the trade numbers were found in the PDF.")
        print("A list of all trade numbers has been saved to not_found_trade_numbers.txt.")
    elif not_found_trade_numbers:
        print("Some trade numbers were not found in the PDF:")
        print("\n".join(not_found_trade_numbers))
    else:
        print("Matched trade numbers and their occurrences in the PDF:")
        for trade_number, occurrences in matched_trade_numbers.items():
            print(f"Trade Number: {trade_number}, Occurrences: {occurrences}")
        print(f"Marked PDF saved as {output_file}.")

if __name__ == "__main__":
    main()
