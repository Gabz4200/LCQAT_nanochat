import slangpy;

// Mul-less ternary GEMV, portable GPU backend (LC-QAT PRD section 5.2).
// One thread per output row; the inner loop resolves activation indices
// through the FP32 LUT (dequantize-on-fetch) and accumulates conditional
// add/sub per ternary weight state. The two asymmetric codebook scales are
// applied once per row - no per-element value multiplications.
// Raw-dispatch convention: first parameter must be `uint3 dispatchThreadID`.
void lcqat_gemv_k3(
    uint3 dispatchThreadID,
    Tensor<float, 1> act_lut,
    Tensor<int, 1> act_indices,
    Tensor<int, 1> weight_indices,
    RWTensor<float, 1> outp,
    int m,
    int n,
    float scale_neg,
    float scale_pos)
{
    int i = (int)dispatchThreadID.x;
    if (i >= m)
        return;
    float pos = 0.0;
    float neg = 0.0;
    int base = i * n;
    for (int j = 0; j < n; ++j)
    {
        float x = act_lut[act_indices[j]];
        int t = weight_indices[base + j];
        if (t == 2)
            pos += x;
        else if (t == 0)
            neg += x;
    }
    outp[i] = scale_pos * pos - scale_neg * neg;
}
