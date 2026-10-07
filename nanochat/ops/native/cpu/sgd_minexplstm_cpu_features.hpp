/**
 * CPU feature detection for SGD-MinExpLSTM kernel.
 * Each kernel has its own copy to avoid dependency conflicts.
 */

#pragma once

#include <cstdint>

namespace sgd_minexplstm_cpu {

class CPUFeatures {
public:
    static bool hasAVX2() {
        static const bool result = detectAVX2();
        return result;
    }

    static bool hasAVX512F() {
        static const bool result = detectAVX512F();
        return result;
    }

    static bool hasAVX512BW() {
        static const bool result = detectAVX512BW();
        return result;
    }

    static bool hasAVX512VL() {
        static const bool result = detectAVX512VL();
        return result;
    }

    static bool hasAVX512BW_VL() {
        return hasAVX512BW() && hasAVX512VL();
    }

    // Full AVX-512 support needed for our kernel
    static bool hasAVX512() {
        return hasAVX512F() && hasAVX512BW() && hasAVX512VL();
    }

private:
    static bool detectAVX2() {
        int cpuInfo[4] = {0};
        // CPUID leaf 7, subleaf 0: EBX bit 5 = AVX2
        __cpuid(cpuInfo, 7);
        bool hwAVX2 = (cpuInfo[1] & (1 << 5)) != 0;

        // Check OS support for YMM registers (XCR0 bit 1 and 2)
        bool osAVX2 = false;
        unsigned long long xcr0 = 0;
#if defined(_MSC_VER)
        xcr0 = _xgetbv(0);
#else
        asm volatile("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
        osAVX2 = (xcr0 & 0x6) == 0x6; // XCR0[1]=1 (XMM) && XCR0[2]=1 (YMM)

        return hwAVX2 && osAVX2;
    }

    static bool detectAVX512F() {
        int cpuInfo[4] = {0};
        // CPUID leaf 7, subleaf 0: EBX bit 16 = AVX-512F
        __cpuid(cpuInfo, 7);
        bool hwAVX512F = (cpuInfo[1] & (1 << 16)) != 0;

        // Check OS support for ZMM registers (XCR0 bits 1,2,5,6,7)
        bool osAVX512 = false;
        unsigned long long xcr0 = 0;
#if defined(_MSC_VER)
        xcr0 = _xgetbv(0);
#else
        asm volatile("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
        osAVX512 = (xcr0 & 0xE6) == 0xE6; // XCR0[1,2,5,6,7]

        return hwAVX512F && osAVX512;
    }

    static bool detectAVX512BW() {
        int cpuInfo[4] = {0};
        // CPUID leaf 7, subleaf 0: EBX bit 30 = AVX-512BW
        __cpuid(cpuInfo, 7);
        bool hwAVX512BW = (cpuInfo[1] & (1 << 30)) != 0;
        return hwAVX512BW;
    }

    static bool detectAVX512VL() {
        int cpuInfo[4] = {0};
        // CPUID leaf 7, subleaf 0: EBX bit 31 = AVX-512VL
        __cpuid(cpuInfo, 7);
        bool hwAVX512VL = (cpuInfo[1] & (1 << 31)) != 0;
        return hwAVX512VL;
    }
};

} // namespace sgd_minexplstm_cpu