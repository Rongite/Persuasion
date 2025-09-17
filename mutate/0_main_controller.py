import os
import subprocess
import argparse

def run_script(script_name, args):
    """
    运行一个 Python 脚本，并传入指定的命令行参数。
    
    :param script_name: 脚本文件名（例如 "scriptA.py")
    :param args: 参数列表（例如 ["--var1", "A1", "--mode", "test"])
    """
    # 构造脚本的完整路径（脚本和控制脚本处于同一目录）
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    print(f"开始运行 {script_name}，参数：{args}")
    try:
        # 使用 subprocess.run 来调用
        # 注意：将 "python"、脚本路径，以及参数列表拼接
        subprocess.run(["python", script_path] + args, check=True)
        print(f"完成运行 {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()
    # 1. 定义三个脚本的运行顺序
    scripts_in_order = ["1_run.py", "2_inference_scaling_all.py", "3_data_statistics.py"]

    # 2. 为每个脚本分别指定参数
    #   - 如果脚本不需要参数，可给空列表 []
    script_args = {
        "1_run.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "2_inference_scaling_all.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "3_data_statistics.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]  # 也可使用非 --key value 形式
    }

    # 3. 依次顺序运行脚本
    for script_name in scripts_in_order:
        args_for_script = script_args.get(script_name, [])
        run_script(script_name, args_for_script)

    print("三个脚本按照指定顺序全部运行完毕。")
