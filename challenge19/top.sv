`timescale 1ns / 1ps

module tb_binary_lif_neuron;

    // Parameters
    parameter THRESHOLD = 4;
    parameter LEAK = 1;
    parameter MAX_MEM = 15;

    // DUT signals
    logic clk, rst, in;
    logic spike_out;

    // Instantiate DUT
    binary_lif_neuron #(
        .THRESHOLD(THRESHOLD),
        .LEAK(LEAK),
        .MAX_MEM(MAX_MEM)
    ) dut (
        .clk(clk),
        .rst(rst),
        .in(in),
        .spike_out(spike_out)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Stimulus task
    task apply_input(string label, logic [31:0] pattern[], int delay_cycles);
        $display("\n=== %s ===", label);
        for (int i = 0; i < pattern.size(); i++) begin
            in = pattern[i];
            #10;  // one clock cycle
            $display("Cycle %0d: in=%0b, spike_out=%0b", i, in, spike_out);
        end
        repeat (delay_cycles) begin
            in = 0;
            #10;
            $display("Cycle %0d: in=0, spike_out=%0b", pattern.size() + delay_cycles, spike_out);
        end
    endtask

    // Test sequence
    initial begin
        // Init
        clk = 0;
        rst = 1;
        in = 0;
        #10;
        rst = 0;

        // Test 1: Constant input below threshold (e.g. 1, but not enough times)
        logic [31:0] test1[] = '{1, 1, 0, 1};  // Membrane will never reach 4
        apply_input("Test 1: Constant input below threshold", test1, 2);

        // Reset
        rst = 1; #10; rst = 0;

        // Test 2: Accumulation to threshold
        logic [31:0] test2[] = '{1, 1, 1, 1};  // Spike expected at end
        apply_input("Test 2: Accumulate until threshold", test2, 2);

        // Reset
        rst = 1; #10; rst = 0;

        // Test 3: No input — leakage to 0
        logic [31:0] test3[] = '{1, 1, 1};  // Integrate then leak
        apply_input("Test 3: Leakage with no input", test3, 5);

        // Reset
        rst = 1; #10; rst = 0;

        // Test 4: Strong continuous input (immediate spiking)
        logic [31:0] test4[] = '{1, 1, 1, 1, 1, 1};  // Should spike repeatedly
        apply_input("Test 4: Strong continuous input", test4, 2);

        $finish;
    end

endmodule

