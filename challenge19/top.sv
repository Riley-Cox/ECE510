module tb_lif_neuron;

    logic clk = 0, rst;
    logic [3:0] input_current;
    logic spike;
    logic [3:0] potential;

    lif_neuron dut (
        .clk(clk),
        .rst(rst),
        .input_current(input_current),
        .spike(spike),
        .potential(potential)
    );

    // Clock generation
    always #5 clk = ~clk;

    task reset();
        rst = 1; input_current = 0;
        @(posedge clk);
        rst = 0;
    endtask

    task check(string test_name, int expected_spike, int expected_pot);
        if (spike !== expected_spike || potential !== expected_pot) begin
            $display("FAIL [%s] @ time %0t: spike=%0b (exp=%0b), pot=%0d (exp=%0d)", 
                test_name, $time, spike, expected_spike, potential, expected_pot);
        end else begin
            $display("PASS [%s] @ time %0t", test_name, $time);
        end
    endtask

    initial begin
        $display("Starting LIF neuron testbench...");
        reset();

        // Test 1: Constant subthreshold input (3 < THRESHOLD=8)
        input_current = 3;
        repeat (5) begin
            @(posedge clk);
            check("Subthreshold input", 0, potential);
        end

        // Test 2: Accumulate to threshold
        reset();
        input_current = 3;
        repeat (2) @(posedge clk);  // Should not spike
        input_current = 4; @(posedge clk);  // Should spike now
        check("Accumulate to threshold", 1, 0);

        // Test 3: Leak with no input
        reset();
        input_current = 5;
        @(posedge clk); // potential = 5 - 1 = 4
        input_current = 0;
        @(posedge clk); // potential = 4 - 1 = 3
        @(posedge clk); // potential = 2
        @(posedge clk); // potential = 1
        @(posedge clk); // potential = 0
        check("Leak with no input", 0, 0);

        // Test 4: Strong input causing immediate spike
        reset();
        input_current = 8; // ≥ THRESHOLD
        @(posedge clk);
        check("Strong input immediate spike", 1, 0);

        $display("All tests completed.");
        $finish;
    end

endmodule

