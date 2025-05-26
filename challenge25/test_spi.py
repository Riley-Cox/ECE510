import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer

@cocotb.test()
async def spi_test(dut):
    # Basic startup
    dut.cs_n.value = 1
    dut.sclk.value = 0
    dut.mosi.value = 0
    dut.rst_n.value = 0
    await Timer(100, units="ns")
    dut.rst_n.value = 1
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await Timer(20, units="ns")

    # Do nothing but finish
    cocotb.log.info("Startup OK")
    await Timer(100, units="ns")

