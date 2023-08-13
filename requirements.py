# 导入pip
import subprocess
import sys
import argparse

# 使用Python类型提示使代码更易读
from typing import List, Optional


def pip_install(proxy: Optional[str], args: List[str]) -> None:
    if proxy is None:
        # 使用subprocess运行pip命令安装包
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"--proxy={proxy}", *args],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="install requirements")
    parser.add_argument("--cuda", default=None, type=str)
    parser.add_argument(
        "--proxy",
        default=None,
        type=str,
        help="specify http proxy, [http://127.0.0.1:1080]",
    )
    args = parser.parse_args()

    pkgs = f"""
    cython
    scikit-image
    loguru
    matplotlib
    tabulate
    tqdm
    pywin32
    PyAutoGUI
    PyYAML>=5.3.1
    opencv_python
    keyboard
    Pillow
    pymouse
    numpy==1.19.5
    torch==1.8.2+{"cpu" if args.cuda is None else "cu" + args.cuda} -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    torchvision==0.9.2+{"cpu" if args.cuda is None else "cu" + args.cuda} --no-deps -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    thop --no-deps
    git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    """

    for line in pkgs.split("\n"):
        # 处理空行中的多个空格
        line = line.strip()

        if len(line) > 0:
            # 使用pip的内部API已被弃用。在将来的pip版本中，这种方法将失败。
            # 最可靠的方法是在子进程中运行pip。
            pip_install(args.proxy, line.split())

    print("\nsuccessfully installed requirements!")


if __name__ == "__main__":
    main()