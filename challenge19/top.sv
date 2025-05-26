module top;

    logic clk = 0;
    logic rst;
    logic input_spike;
    logic spike;
    logic [7:0] potential;

    // Instantiate the neuron
    lif_neuron_fixedpoint #(
        .THRESHOLD(8'd60),       // 3.75
        .LEAK_FACTOR(8'd12)      // 0.75
    ) dut (
        .clk(clk),
        .rst(rst),
        .input_spike(input_spike),
        .spike(spike),
        .potential(potential)
    );

    // Clock generation
    always #5 clk = ~clk;

    task reset();
        rst = 1;
        input_spike = 0;
        @(posedge clk);
        rst = 0;
    endtask

    task tick(input logic in);
        input_spike = in;
        @(posedge clk);
        $display("Time %0t | In=%0b | Spike=%0b | Pot=%0d (Q4.4 = %0f)",
                 $time, in, spike, potential, potential / 16.0);
    endtask

    initial begin
        $display("Starting fixed-point LIF neuron testbench...");
        reset();

        // Test 1: No input, just leak
        $display("\nTest 1: No input (leak test)");
        repeat (10) tick(0);

        // Test 2: Accumulating input to cause spike
        $display("\nTest 2: Accumulate to threshold");
        repeat (15) tick(1);  // Should spike near 4.0

        // Test 3: Constant input post-spike reset
        $display("\nTest 3: Verify reset after spike");
        repeat (15) tick(1);

        // Test 4: Leak after single spike input
        $display("\nTest 4: One input then decay");
        tick(1);  // Add one input
        repeat (15) tick(0);  // Leak down

        // Test 5: Strong single spike (force threshold manually)
        $display("\nTest 5: Force immediate spike (manually load potential)");
        reset();
        force dut.potential = 8'd64;  // Set to threshold
        tick(0);  // Should spike immediately
        release dut.potential;

        $display("\nAll tests completed.");
        $finish;
    end

endmodule

