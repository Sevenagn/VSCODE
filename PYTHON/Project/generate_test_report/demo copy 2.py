import win32com.client

def get_tasks():
    scheduler = win32com.client.Dispatch('Schedule.Service')
    scheduler.Connect()

    root_folder = scheduler.GetFolder('\\')
    tasks = root_folder.GetTasks(0)

    for task in tasks:
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

if __name__ == "__main__":
    get_tasks()
