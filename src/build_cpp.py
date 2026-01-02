import os
import subprocess
import shutil

# ==========================================
# C++ 策略源码 (内嵌在脚本里，防丢失)
# ==========================================
CPP_SOURCE_CODE = """
#include <stdio.h>

extern "C" {
    /**
     * C++ 策略生成器 (Exported Function)
     * 简单的阈值判断策略
     */
    __declspec(dllexport) void generate_signals(double* predictions, double* signals, int length, double threshold) {
        for (int i = 0; i < length; i++) {
            if (predictions[i] > threshold) {
                signals[i] = 1.0;  // Buy
            } else {
                signals[i] = 0.0;  // Hold/Sell
            }
        }
    }
}
"""

def build_dll():
    # ==========================================
    # 1. 核心修复：自动定位到项目根目录
    # ==========================================
    # 获取当前脚本文件 (src/build_cpp.py) 的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取 src 目录路径
    src_dir = os.path.dirname(current_script_path)
    # 获取项目根目录 (Quant_System)
    project_root = os.path.dirname(src_dir)
    
    # 强制切换工作目录到项目根目录
    os.chdir(project_root)
    print(f">>> [Setup] Working Directory switched to: {os.getcwd()}")

    # ==========================================
    # 2. 清理环境
    # ==========================================
    if os.path.exists("build"):
        try:
            shutil.rmtree("build")
            print(">>> [Clean] Removed old build directory.")
        except:
            print(">>> [Warning] Could not clean build directory, continuing...")

    # ==========================================
    # 3. 检查/生成源文件
    # ==========================================
    if not os.path.exists("strategy.cpp"):
        print(">>> [Fix] strategy.cpp not found. Re-creating it in root...")
        with open("strategy.cpp", "w") as f:
            f.write(CPP_SOURCE_CODE)
    else:
        print(">>> [Check] strategy.cpp found.")

    # ==========================================
    # 4. 生成 CMakeLists.txt
    # ==========================================
    print(">>> [Config] Creating CMakeLists.txt ...")
    cmake_content = """cmake_minimum_required(VERSION 3.10)
project(QuantStrategy)
add_library(strategy SHARED strategy.cpp)
"""
    with open("CMakeLists.txt", "w") as f:
        f.write(cmake_content)

    # ==========================================
    # 5. 调用 CMake 编译
    # ==========================================
    print("\n>>> [CMake] Configuring ...")
    # -S . 表示源码在当前目录(根目录), -B build 表示构建文件放 build 目录
    ret = subprocess.call(["cmake", "-S", ".", "-B", "build"])
    if ret != 0:
        print(">>> [Error] CMake Configuration failed.")
        return

    print("\n>>> [Compile] Building ...")
    ret = subprocess.call(["cmake", "--build", "build", "--config", "Release"])
    if ret != 0:
        print(">>> [Error] Compilation failed.")
        return

    # ==========================================
    # 6. 搬运 DLL 到根目录
    # ==========================================
    print("\n>>> [Install] Looking for strategy.dll ...")
    dll_name = "strategy.dll"
    possible_paths = [
        os.path.join("build", "Release", dll_name),
        os.path.join("build", "Debug", dll_name),
        os.path.join("build", dll_name)
    ]
    
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            # 复制到项目根目录
            target_path = os.path.join(".", dll_name)
            shutil.copy(path, target_path)
            print(f">>> [SUCCESS] DLL deployed to: {os.path.abspath(target_path)}")
            found = True
            break
            
    if not found:
        print(">>> [Error] Build finished but DLL not found in build folder.")
    else:
        print("\n=== C++ BUILD & DEPLOY COMPLETE ===")

if __name__ == "__main__":
    try:
        build_dll()
    except Exception as e:
        print(f"Fatal Error: {e}")