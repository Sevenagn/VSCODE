import wmi

def get_process_cpu_usage():
    try:
        # 连接到本地计算机
        c = wmi.WMI()
        # 获取所有进程的性能数据
        processes = c.Win32_PerfFormattedData_PerfProc_Process()
        # 存储进程信息的列表
        process_list = []
        for process in processes:
            process_info = {
                "ProcessId": process.IDProcess,
                "Name": process.Name,
                "CPUUsage": process.PercentProcessorTime
            }
            process_list.append(process_info)
        return process_list
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    processes = get_process_cpu_usage()
    if processes:
        print("本机的进程CPU使用率:")
        for proc in processes:
            print(f"进程ID: {proc['ProcessId']}, 名称: {proc['Name']}, CPU使用率: {proc['CPUUsage']}%")
    else:
        print("无法获取本机的进程信息。")
