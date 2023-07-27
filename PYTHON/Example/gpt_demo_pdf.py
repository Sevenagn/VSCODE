import fitz  # PyMuPDF

def highlight_trade_numbers(input_path, output_path, trade_numbers, page_number, highlight_color=(1, 0, 0)):
    pdf_document = fitz.open(input_path)
    page = pdf_document[page_number]
    
    for trade_number in trade_numbers:
        occurrences = page.search_for(trade_number)
        for rect in occurrences:
            # 添加高亮标注
            highlight = page.add_highlight_annot(rect)
            # 设置高亮颜色
            highlight.set_colors(stroke=highlight_color)
    
    pdf_document.save(output_path, garbage=4, deflate=True)
    pdf_document.close()

if __name__ == "__main__":
    input_file = r"D:\wechatpayinfo\2023.pdf"   # 输入的PDF文件路径
    output_file = r"D:\wechatpayinfo\2023_new.pdf"   # 输出的高亮后的PDF文件路径
    page_number = 0            # 要添加高亮的页面索引（从0开始）
    highlight_color = (1, 0, 0)  # 高亮颜色，这里使用红色

    # 定义要高亮的交易单号列表
    trade_numbers = ["4200001840202305248766"]

    # 根据交易单号在PDF中添加高亮
    highlight_trade_numbers(input_file, output_file, trade_numbers, page_number, highlight_color)

    print(f"Trade numbers {', '.join(trade_numbers)} highlighted in page {page_number} of the PDF.")
