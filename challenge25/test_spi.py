import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge

class SPIMaster:
    def __init__(self, dut, half_period_ns=5):
        self.dut = dut
        self.half = half_period_ns

    async def transfer(self, byte_out: int) -> int:
        recv = 0
        self.dut.cs_n.value = 0
        await Timer(1, units="ns")

        for i in range(8):
            self.dut.mosi.value = (byte_out >> (7 - i)) & 1

            self.dut.sclk.value = 0
            await Timer(self.half, units="ns")

            self.dut.sclk.value = 1
            await Timer(self.half, units="ns")

            bit_val = self.dut.miso.value.binstr
            recv = (recv << 1) | (int(bit_val) if bit_val in ('0', '1') else 0)

        self.dut.sclk.value = 0
        self.dut.cs_n.value = 1
        return recv

@cocotb.test()
async def spi_test(dut):
    # Initialize
    dut.cs_n.value = 1
    dut.sclk.value = 0
    dut.mosi.value = 0
    dut.rst_n.value = 0
    await Timer(100, units="ns")
    dut.rst_n.value = 1

    # Start system clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await RisingEdge(dut.clk)

    spi = SPIMaster(dut)

    # Test case(s)
    test_vectors = [
        (0x3C, 0xA5),
        (0x00, 0x5A)
    ]

    for i, (send, expected) in enumerate(test_vectors):
        dut.tx_data.value = expected
        await RisingEdge(dut.clk)
        await Timer(20, units="ns")

        recv = await spi.transfer(send)
        cocotb.log.info(f"[Test {i}] Sent: 0x{send:02X}, Received: 0x{recv:02X}, Expected: 0x{expected:02X}")
        assert recv == expected, f"[Test {i}] FAIL: Sent 0x{send:02X}, Expected 0x{expected:02X}, Got 0x{recv:02X}"

