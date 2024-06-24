import wmi

def get_remote_tasks(remote_computer_name):
    try:
        # 连接到远程计算机
        c = wmi.WMI()
        # 获取 Task Scheduler 的根文件夹
        root_folder = c.Win32_ScheduledJob()
        # 存储任务信息的列表
        task_list = []
        for task in root_folder:
            task_info = {
                "Name": task.Name,
                "Command": task.Command,
                "Status": task.Status
            }
            task_list.append(task_info)
            print(task)
            print('Task Name:', task.Name)
            print('Path:', task.Path)
            print('Enabled:', task.Enabled)
            print('Last Task Result:', task.LastTaskResult)
            print('Last Run Time:', task.LastRunTime)
            print('Next Run Time:', task.NextRunTime)
            task_def = task.Definition
            actions = task_def.Actions
            for action in actions:
                if action.Type == 0:  # 0 represents "Executes a program"
                    print('Program Path:', action.Path)
            print('----------------------------------------')
        return task_list
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    remote_computer_name = "REMOTE_COMPUTER_NAME"  # 替换为远程计算机的名称或 IP 地址

    tasks = get_remote_tasks(remote_computer_name)
    if tasks:
        print(f"{remote_computer_name} 上的计划任务:")
        for task in tasks:
            print(f"名称: {task['Name']}, 命令: {task['Command']}, 状态: {task['Status']}")
    else:
        print(f"无法获取 {remote_computer_name} 上的计划任务。")
