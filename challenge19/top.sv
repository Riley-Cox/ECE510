`timescale 1ns / 1ps

module top;

    // Parameters
    parameter THRESHOLD = 4;
    parameter LEAK = 1;
    parameter MAX_MEM = 15;

    // DUT signals
    logic clk, rst, in;
    logic spike_out;

    // Test vectors and expected outputs
    logic [31:0] test1[]       = '{1, 1, 0, 1};
    logic [31:0] expected1[]   = '{0, 0, 0, 0};

    logic [31:0] test2[]       = '{1, 1, 1, 1};
    logic [31:0] expected2[]   = '{0, 0, 0, 1};

    logic [31:0] test3[]       = '{1, 1, 1, 0, 0, 0};
    logic [31:0] expected3[]   = '{0, 0, 0, 0, 0, 0};

    logic [31:0] test4[]       = '{1, 1, 1, 1, 1, 1, 1, 1};
    logic [31:0] expected4[]   = '{0, 0, 0, 1, 0, 0, 0, 1};

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

    // Error-checking input application
    task automatic apply_input(
        string label,
        logic [31:0] pattern[],
        logic [31:0] expected_spikes[],
        int delay_cycles
    );
        int errors = 0;
        $display("\n=== %s ===", label);
        for (int i = 0; i < pattern.size(); i++) begin
            in = pattern[i];
            #10;
            $display("Cycle %0d: in=%0b, spike_out=%0b (expected %0b)", i, in, spike_out, expected_spikes[i]);
            if (spike_out !== expected_spikes[i]) begin
                $display("  ERROR: Mismatch at cycle %0d", i);
                errors++;
            end
        end
        repeat (delay_cycles) begin
            in = 0;
            #10;
        end
        if (errors == 0)
            $display("  PASS: %s\n", label);
        else
            $display("  FAIL: %0d mismatches in %s\n", errors, label);
    endtask

    // Test sequence
    initial begin
        clk = 0;
        rst = 1;
        in = 0;
        #10;
        rst = 0;

        apply_input("Test 1: Constant input below threshold", test1, expected1, 2);
        rst = 1; #10; rst = 0;

        apply_input("Test 2: Accumulate until threshold", test2, expected2, 2);
        rst = 1; #10; rst = 0;

        apply_input("Test 3: Leakage with no input", test3, expected3, 2);
        rst = 1; #10; rst = 0;

        apply_input("Test 4: Strong continuous input", test4, expected4, 2);

        $display("=== All tests completed ===");
        $finish;
    end

endmodule

