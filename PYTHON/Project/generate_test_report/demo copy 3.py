import psutil

def get_local_processes():
    # 获取所有进程列表
    all_processes = psutil.process_iter()
    
    # 存储进程信息的列表
    processes_info = []
    
    # 遍历每个进程并获取信息
    for proc in all_processes:
        try:
            # 获取进程信息
            process_info = {
                "pid": proc.pid,
                "name": proc.name(),
                "exe": proc.exe()
            }
            # 将进程信息添加到列表中
            processes_info.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 处理进程不存在、权限拒绝或僵尸进程的情况
            pass
    
    return processes_info

if __name__ == "__main__":
    local_processes = get_local_processes()
    print("本机进程信息:")
    for proc in local_processes:
        print(f"进程ID: {proc['pid']}, 名称: {proc['name']}, 执行路径: {proc['exe']}")
