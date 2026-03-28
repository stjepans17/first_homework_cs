# Copyright (c) 2022 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from gem5.isas import ISA
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor

from m5.objects import RiscvMinorCPU
from m5.objects.FuncUnitConfig import *
from m5.objects.BranchPredictor import (
    TournamentBP,
    MultiperspectivePerceptronTAGE64KB,
    LocalBP
)
from m5.objects.BaseMinorCPU import MinorFUTiming, MinorFU, MinorFUPool
from m5.objects.BaseMinorCPU import minorMakeOpClassSet
from m5.objects.BaseMinorCPU import MinorDefaultIntFU, MinorDefaultIntMulFU, MinorDefaultIntDivFU, MinorDefaultMemFU


# Custom functional unit definition

class MinorCustomFloatALU(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatAdd",
            "FloatCmp",
            "FloatCvt",
            "FloatMisc"
        ]
    )
    timings = [MinorFUTiming(description="FloatALUCustom", srcRegsRelativeLats=[2])]
    opLat = 6

class MinorCustomFloatMult(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatMult",
            "FloatMultAcc" # multiply-accumulate instruction
        ]
    )
    timings = [MinorFUTiming(description="FloatMultCustom", srcRegsRelativeLats=[2])]
    opLat = 6


class MinorCustomFloatDiv(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatDiv",
            "FloatSqrt"
        ]
    )
    timings = [MinorFUTiming(description="FloatDivCustom", srcRegsRelativeLats=[2])]
    opLat = 6



class MyCustomFUPool(MinorFUPool):
            funcUnits = [
                MinorDefaultIntFU(), # default integer ALU
                MinorDefaultIntFU(), # default integer ALU
                MinorDefaultIntMulFU(), # default integer multiplier
                MinorDefaultIntDivFU(),  # default integer divider
                MinorCustomFloatALU(),  # custom floating-point ALU
                MinorCustomFloatMult(), # custom floating-point multiplier
                MinorCustomFloatDiv(),  # custom floating-point divider
                MinorDefaultMemFU(), # default memory access FU
            ]


#RiscvMinorCPU is one of gem5's internal models that implements an in order pipeline with RISCV ISA. 
# You can find the details in gem5/src/arch/riscv/RiscvCPU.py. 

# RiscvMinorCPU extends BaseMinorCPU with RISCV ISA
# the implements an in order pipeline. Please refer to
#   https://www.gem5.org/documentation/general_docs/cpu_models/minor_cpu
# python definition: gem5/src/minor/BaseMinorCPU.py
# to learn more about BaseMinorCPU.


# Functional units latencies. In order:
# - Integer ALU and Memory access address calculation
# - Integer Multiplication
# - Integer Division
# - Float ALU
# - Float Multiplication
# - Float Division

# The latency specifies the number of cycles it takes for the
# functional unit to execute an instruction after it is issued.
# The INTEGER_ALU_LATENCY controls also the latency of the memory
# address calculation
INTEGER_ALU_LATENCY = 1
INTEGER_MUL_LATENCY = 3
INTEGER_DIV_LATENCY = 3
FLOAT_ALU_LATENCY = 3
FLOAT_MUL_LATENCY = 6
FLOAT_DIV_LATENCY = 12

# The issue latency is the number of cycles until another instruction can be issued 
# to the functional unit after an instruction has already been issued.
# The INTEGER_ALU_ISSUE_LATENCY controls also the issue latency of the memory
# address calculation
INTEGER_ALU_ISSUE_LATENCY = 0
INTEGER_MUL_ISSUE_LATENCY = 0
INTEGER_DIV_ISSUE_LATENCY = 0
FLOAT_ALU_ISSUE_LATENCY = 0
FLOAT_MUL_ISSUE_LATENCY = 0
FLOAT_DIV_ISSUE_LATENCY = 0

 


class InOrdCPUCore(RiscvMinorCPU):
    def __init__(self):

        """
        Configures the following CPU subsystems:
        Functional Unit Latencies:
            - Operation latencies (opLat): The number of cycles required for each functional unit
              to complete an operation:
                - Integer ALU, Multiplier, Divider
                - Floating-point ALU, Multiplier, Divider
            - Issue latencies (issueLat): The number of cycles that must elapse before another
              instruction can be issued to the same functional unit after an instruction has been issued.
        Functional Unit Timings:
            - Defines MinorFUTiming objects that specify operation latencies and source register
              relative latencies for different operation classes (IntAlu, MemRead/Write, FloatMult, etc.).
            - srcRegsRelativeLats: Relative latency of source registers for each operation class.
            - extraAssumedLat and extraCommitLat: Additional assumed or commit latencies.
            - cantForwardFromFUIndices: Specifies which functional units cannot forward results
              to the current unit, preventing data forwarding from specific sources.
        Pipeline Configuration:
            - executeInputWidth: Instructions sent to execute stage per cycle (1).
            - executeInputBufferSize: Buffer size between issue and execute stages (1).
            - decodeInputBufferSize: Buffer size between decode and issue stages (1).
            - executeIssueLimit: Maximum instructions dispatched per cycle (2).
            - executeMemoryIssueLimit: Maximum memory operations dispatched per cycle (1).
            - decodeToExecuteForwardDelay: Forwarding delay between decode and execute stages (1 cycle).
        CPU Behavior:
            - enableIdling: Disabled (False) - CPU does not idle when no instructions are available.
            - fetch1LineWidth: Instruction fetch width set to 512 bits.
            - branchPred: Tournament branch predictor for branch prediction.
        
        Initialize an in-order CPU configuration with specified resources.

        Args:
            width (int): The fetch/decode/execute width of the CPU pipeline.
            num_fp_regs (int): The number of floating-point registers.
            num_int_regs (int): The number of integer registers.

        This method configures:
        - Functional unit latencies (operation latency) for integer and floating-point ALUs, 
          multipliers, and dividers
        - Issue latencies for each functional unit type
        - Load and store queue entry counts (128 each)

        """
        super().__init__()
        
        # Create custom FU pool:
        # - Integer ALU and 
        # - Integer Multiplication
        # - Integer Division
        # - Float ALU
        # - Float Multiplication
        # - Float Division
        # - Memory access FU (address calculation and load/store)
        self.executeFuncUnits = MyCustomFUPool()
        

        # The parameter opLat is the latency of the functional unit, i.e., the number of cycles it takes for the
        self.executeFuncUnits.funcUnits[0].opLat = INTEGER_ALU_LATENCY
        self.executeFuncUnits.funcUnits[1].opLat = INTEGER_ALU_LATENCY
        self.executeFuncUnits.funcUnits[2].opLat = INTEGER_MUL_LATENCY
        self.executeFuncUnits.funcUnits[3].opLat = INTEGER_DIV_LATENCY
        self.executeFuncUnits.funcUnits[4].opLat = FLOAT_ALU_LATENCY
        self.executeFuncUnits.funcUnits[5].opLat = FLOAT_MUL_LATENCY
        self.executeFuncUnits.funcUnits[6].opLat = FLOAT_DIV_LATENCY
        self.executeFuncUnits.funcUnits[7].opLat = INTEGER_ALU_LATENCY # Memory access latency is the same as integer ALU latency
        # The parameter issueLat controls the issue latency of the functional unit, i.e., the number of cycles
        # until another instruction can be issued to the functional unit after an instruction has already been issued.
        self.executeFuncUnits.funcUnits[0].issueLat = INTEGER_ALU_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[1].issueLat = INTEGER_ALU_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[2].issueLat = INTEGER_MUL_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[3].issueLat = INTEGER_DIV_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[4].issueLat = FLOAT_ALU_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[5].issueLat = FLOAT_MUL_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[6].issueLat = FLOAT_DIV_ISSUE_LATENCY
        self.executeFuncUnits.funcUnits[7].issueLat = INTEGER_ALU_ISSUE_LATENCY # Memory access issue latency is the same as integer ALU issue latency


        # The parameter cantForwardFromFUIndices specifies the indices of the functional units from which the
        # functional unit cannot forward results.

        # Pipeline configuration parameters
        # The number of instructions that can be sent to the execute stage per cycle
        self.executeInputWidth = 1
        # The number of instructions that can be buffered between the issue and execute stages
        self.executeInputBufferSize = 4 
        # The number of instructions that can be buffered between the decode and issue stages
        self.decodeInputBufferSize = 4
        # Issue limits for instruction dispatch
        self.executeIssueLimit = 1
        self.executeMemoryIssueLimit = 1
        
        # Pipeline stage delays
        self.decodeToExecuteForwardDelay = 1
        
        # CPU behavior settings
        self.enableIdling = False
        
        # Instruction fetch width (in bytes)
        self.fetch1LineWidth = 64 
        




# Along with BaseCPUCore, CPUStdCore wraps CPUCore to a core compatible
# with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_core.py
# to learn more about BaseCPUCore.


class InOrdCPUStdCore(BaseCPUCore):
    def __init__(self):
        """
        CPU compatible with gem5's standard library that wraps the InOrdCPUCore, which is an in-order CPU model based on RiscvMinorCPU. This class serves as a bridge between the InOrdCPUCore and the standard library's BaseCPUCore, allowing it to be used in a wider range of simulations and configurations.
        """
        core = InOrdCPUCore()
        super().__init__(core, ISA.RISCV)



# InOrdCPU along with BaseCPUProcessor wraps CPUCore to a processor
# compatible with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_processor.py
# to learn more about BaseCPUProcessor.


class InOrdCPU(BaseCPUProcessor):
    def __init__(self):
        """
        Processor compatible with gem5's standard library that wraps the InOrdCPUStdCore,
        which is an in-order CPU model based on RiscvMinorCPU. 
        This class serves as a bridge between the InOrdCPUStdCore and the standard 
        library's BaseCPUProcessor, allowing it to be used in a wider range of 
        simulations and configurations.
        """
        cores = [InOrdCPUStdCore()]
        super().__init__(cores)

    def get_area_score(self):
        """
        Does not matter.  
        """
        score = 0
        return score

class RiscV_InOrder_CPU(InOrdCPU):
    def __init__(self):
        super().__init__()

