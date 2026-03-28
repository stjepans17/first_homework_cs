from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import PrivateL1CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import obtain_resource
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import CustomResource
from cpuO3_model import RISCV_O3_CPU
from cpuInORD_model import RiscV_InOrder_CPU

cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB")


memory = SingleChannelDDR3_1600("7GiB")

#Source code for simple processor gem5/src/python/gem5/components/processors/simple_processor.py
processor = RISCV_O3_CPU()


board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)



# Set the workload.
binary = CustomResource("../workload/scaled_dot_product.bin")
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()