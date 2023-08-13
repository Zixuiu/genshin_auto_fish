#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 上面两行是为了确保脚本在任何环境下都能正确执行

# 版权声明
# 该脚本是由 Megvii, Inc. 及其关联公司拥有版权，保留所有权利

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension

# 检查 PyTorch 版本是否符合要求
torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

# 获取扩展模块的信息
def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "yolox", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "yolox._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

# 从 __init__.py 文件中获取版本号
with open("yolox/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)

# 从 README.md 文件中获取长描述
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 设置包的信息
setuptools.setup(
    name="yolox",
    version=version,
    author="basedet team",
    python_requires=">=3.6",
    long_description=long_description,
    ext_modules=get_extensions(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
)
