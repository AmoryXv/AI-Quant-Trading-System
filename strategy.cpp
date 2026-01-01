// strategy.cpp
// 这是一个符合 C 标准接口的函数，方便 Python 调用
// extern "C" 告诉编译器不要对函数名进行修饰(Name Mangling)

#include <stdio.h>

extern "C" {
    /**
     * 极简策略生成器
     * * @param predictions  输入数组：模型预测的分数 (指针)
     * @param signals      输出数组：生成的交易信号 (指针，1代表买，0代表空)
     * @param length       数据长度
     * @param threshold    买入阈值 (比如预测收益率 > 0.01 才买)
     */
    __declspec(dllexport) void generate_signals(double* predictions, double* signals, int length, double threshold) {
        
        for (int i = 0; i < length; i++) {
            if (predictions[i] > threshold) {
                signals[i] = 1.0;  // Buy Signal
            } else {
                signals[i] = 0.0;  // No Position
            }
        }
    }
}