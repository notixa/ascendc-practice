#pragma once

/**
 * 本头文件仅供本地 Zed (clangd LSP) 语法分析与代码补全使用，
 * 不参与板端 BiSheng 真实编译。
 */
namespace AscendC {
    int printf(const char* format, ...);
    void InitSocState();
}
