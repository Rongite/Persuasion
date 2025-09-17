import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_script(script_path, args):
    """
    Run a Python script (with absolute or relative path) and pass in arguments.
    :param script_path: Full path to the script file
    :param args: List of arguments, e.g. ["--var1", "A1", ...]
    """
    print(f"开始运行脚本: {script_path}，传入参数: {args}")
    try:
        subprocess.run(["python", script_path] + args, check=True)
        print(f"完成运行脚本: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"运行脚本: {script_path} 出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()
    # 定义脚本目录，比如 Project/scripts
    scripts_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/inference_scaling"

    # 定义脚本名称列表
    script_names = ["1_zero_shot_with_judge_inference_scaling.py", "2_zero_shot_without_judge_inference_scaling.py", "3_few_shots_with_judge_inference_scaling.py", "4_few_shots_without_judge_inference_scaling.py"]

    # 拼接脚本完整路径
    # 如果脚本需要参数，则可在此处或通过字典指定
    scripts_info = {
        "1_zero_shot_with_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "1_zero_shot_with_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "2_zero_shot_without_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "2_zero_shot_without_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "3_few_shots_with_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "3_few_shots_with_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "4_few_shots_without_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "4_few_shots_without_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        }
    }

    # 用并行的方式运行四个脚本
    with ThreadPoolExecutor() as executor:
        futures = []
        for script_name in script_names:
            script_path = scripts_info[script_name]["path"]
            script_args = scripts_info[script_name]["args"]
            futures.append(executor.submit(run_script, script_path, script_args))

        # 等待所有并行任务完成
        for f in futures:
            f.result()

    print("所有脚本已运行完毕。")
