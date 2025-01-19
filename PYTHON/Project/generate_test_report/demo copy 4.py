import wmi
 
def list_wmi_classes():
    try:
        # 连接到本地计算机
        c = wmi.WMI()
        b = c.classes
        d = b.list
        for e in d:
            print(e)
        # print(b)

        # 获取 Win32_ComputerSystem 类
        computer_system = c.Win32_ComputerSystem()[0]
        # 获取与 Win32_ComputerSystem 关联的所有 WMI 类
        classes = computer_system.associators(wmi_result_class="meta_class")
        # 打印所有类名
        for class_info in classes:
            print(class_info.Path_.Class)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_wmi_classes()
