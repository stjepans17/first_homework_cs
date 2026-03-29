# First homework


## Task 1: Analyze performance of O3 processor (5 points)

You are given a program that computes **scaled dot-product attention scores** — the core non-GEMM operation in Transformer models. The source code is located in the `workload/scaled_dot_product.c` file.

To ensure consistent and comparable results across all submissions, all simulations must use the explicitly specified functional unit and cache configuration defined below. These parameters are fixed and must not be changed unless a task explicitly instructs you to vary them. The defult configuration of O3CPU and MinorCPU are definied in python files in folder `default/` and you can refer to them for more details.

### Task 1a:
First, report the following metrics from `stats.txt` for both CPU models:

| Metric | MinorCPU | O3CPU |
|---|---|---|
| CPI | 1.4253 | 0.80843 |
| Total cycles | 3127870 | 1774101 |
| IPC | 0.7016 | 1.236972 |

What is the IPC speedup of O3CPU over MinorCPU and explain it.

1.236972 / 0.7016 = 1.76 => O3CPU has am IPC speedup of around 76% over MinorCPU.

Main reason is that O3CPU supports out-of-order execution, while MinorCPU has to do entire execution sequentially (in-order).

### Task 1b: 
For parameter `SEQ_LEN=64` of workload, run O3CPU with ROB sizes of 16, 32, 64, 128 and 256 entries and Plot IPC vs. ROB size. At what ROB size does performance saturate? What does this tell you about the instruction-level parallelism (ILP) available in this workload?

| Rob size   | 16 | 32 | 64 | 128 | 256 | 
|---|---|---|---|---|---|
| O3 | 0.905859 | 1.088491 | 1.238386 | 1.236972 | 1.236972 |

Perfomance saturates at ROB_SIZE = 64. Going with bigger ROB_SIZE does not show improvement.

It means there are cca 64 instructions worth of independant work, that ROB_SIZE is enough to capture all parallelism needed.


### Task 1c:  
Rerun both the original and optimized version (`scaled_dot_product_adv`) of the attention kernel on O3CPU, sweeping pipeline width  over  2, 4, and 8 width of different stages to the same value. Record IPC for each combination and plot both versions on the same graph as IPC vs. pipeline width. Analyze whether the IPC gap between the two versions grows or shrinks as the pipeline widens, and explain what this reveals about the relationship between  ILP and hardware utilization.

| Width | Original | Optimized | Gap | 
|---|---|---|---|
| 2 | 1.2384 | 1.3796 | 0.1412
| 4 | 1.2943 | 1.4985 | 0.2042
| 8 | 1.3042 | 1.6171 | 0.3129

IPC gap between the 2 version grow as pipeline widens.
The wider pipeline is then the optimized code "improves" more as it can fill those extra issue slots more effectively than the original code.


### Task 1d:
For both version, original and optimized, sweep the number of physical integer and floating-point registers over 64, 96, and 128 entries measuring IPC and the number of stalls caused by register file exhaustion (`system.cpu.rename.fullRegistersEvents`). Plot IPC and stalls vs. number of registers. At what point do additional registers stop improving IPC? Set the pipeline width to 2 and ROB size to 128 for this experiment. 

*Note: if you cannot find the `fullRegistersEvents` stat, than means that is equal to 0 , and you can report 0 for all of them.*

Original (`scaled_dot_product`):

| Metric | 64 | 96 | 128 |
|---|---|---|---|
| IPC | 1.1434 | 1.2809 | 1.2640 |
| Stalls | 7496 | 4181 | 0 |

Optimized (`scaled_dot_product_adv`):

| Metric | 64 | 96 | 128 |
|---|---|---|---|
| IPC | 1.3784 | 1.6150 | 1.5496 |
| Stalls | 2718 | 27 | 0 |

Additional registers stop improving IPC at cca 96 entries.

## Task 2: Branch Prediction and Speculative Execution in Masked Attention (O3CPU) (5 points)

In autoregressive Transformer models (such as GPT), the attention mechanism applies a **causal mask** to prevent each token from attending to future positions. Before the softmax is computed, all positions $j > i$ for query $i$ are set to $-\infty$, ensuring their contribution after softmax is zero. This masking introduces **data-dependent branches** whose taken/not-taken ratio changes systematically across query positions — early queries mask most of the sequence, late queries mask almost nothing. This makes the workload particularly interesting for branch prediction analysis: the branch behavior is neither fully predictable nor fully random, but shifts gradually across the outer loop iterations.

---

### The Workload

The program computes masked scaled dot-product attention for every query position in a sequence. It is located in the `workload/` folder. The two branches of interest are:

```c
void masked_softmax(float *scores, float *output, int query_pos, int len) {

    // Branch 1 — Causal mask
    // Taken ratio = (len - query_pos) / len
    // Shifts from ~100% (query 0) to ~0% (query SEQ_LEN-1)
    for (int j = 0; j < len; j++) {
        if (j > query_pos)
            scores[j] = NEG_INF;
    }

    // Branch 2 — Max reduction
    // Fully data-dependent, no learnable pattern
    float max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val)
            max_val = scores[i];
    }

    // ... exp, sum, normalize (no branches)
}
```

---



### Task 2a
Compile and run the masked attention workload using the baseline configuration. Record the following from `stats.txt`:

| Metric | Value |
|---|---|
| Total instructions committed | 2194513 |
| Total cycles | 1774387 |
| IPC | 1.236773 |
| Branch instructions committed | 335874 |
| Branch mispredictions | 499 |



### Task 2b
Run the workload with the following four branch predictors (`TournamentBP`,`LocalBP`,  `BiModeBP` and `TAGE`), keeping all other parameters fixed. Record for each predictor:

| Predictor | Branch mispredictions  | IPC | 
|---|---|---|
| TAGE | 499 | 1.236773 |
| LocalBP | 5430 | 1.203859 |
| Tournament | 4673 | 1.214296 |
| BimodeBP | 4788 | 1.213011 |

*Note: Metric for branch mispredictions count is: `branchPred.condIncorrect` while the metric for Branch instructions committed is: `branchPred.committed_0::total`*


Which predictor achieves the lowest misprediction count and why?

**TAGE** achieves the lowest misprediction count with 499 mispredictions. It uses multiple predictor tables indexed by different history lengths, allowing it to capture both the gradual shift in the causal mask branch and short-term variations through multi-scale pattern matching.

Why others are worse:
- **BiModeBP**: Uses only simple 2-bit saturating counters indexed by branch address therefore, it can't track evolving patterns because it treats each branch statically and can't adapt to the gradually changing taken ratio.
- **LocalBP**: Uses only local branch history (recent outcomes of the same branch). With this it struggles with the global systematic shift of the causal mask pattern across query positions.
- **Tournament**: Combines global and local history selectors, but lacks TAGE's geometric hierarchy. Whit its setup it can't efficiently capture patterns at multiple timescales like TAGE's layered tables.

### Task 2c
Using the best-performing predictor from *Q2*, sweep ROB size over **32, 64, and 128 entries**:

Our best one was TAGE.

Record for each ROB size:

| ROB Size | Misprediction count | IPC | Squashed instruction count |
|---|---|---|---|
| 32 | 501 | 1.088813 | 513 |
| 64 | 496 | 1.238022 | 469 |
| 128 | 499 | 1.236773 | 442 |

*Note: Metric for Squashed instruction count is: "core.numSquashedInsts"*

As ROB size increases, what happens simultaneously to IPC and to instructions flushed per misprediction? .

As ROB size increases from 32 to 64, IPC improves significantly while squashed instructions decrease. The larger ROB enables more instruction level parallelism and reduces the pipeline depth, so fewer dependent instructions are flushed per misprediction. Beyond ROB 64, IPC saturates but squashed instructions continue declining, indicating that while performance stops improving, recovery from mispredictions remains more efficient in the larger ROB due to reduced window size relative to in-flight instructions.