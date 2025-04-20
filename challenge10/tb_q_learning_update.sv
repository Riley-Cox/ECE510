// tb_q_learning_update.sv

`timescale 1ns/1ps

module tb_q_learning_update;

    // Parameters
    localparam ROWS = 5;
    localparam COLS = 5;
    localparam ACTIONS = 4;
    localparam ADDR_WIDTH = 7;
    localparam DATA_WIDTH = 8;

    // Inputs
    logic clk;
    logic rst;
    logic [2:0] row;
    logic [2:0] col;
    logic [1:0] action;
    logic [2:0] next_row;
    logic [2:0] next_col;
    logic [7:0] reward;

    // Outputs
    logic [ADDR_WIDTH-1:0] addr_sa;
    logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1];

    // Instantiate the DUT
    q_learning_update #(
        .ROWS(ROWS),
        .COLS(COLS),
        .ACTIONS(ACTIONS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .row(row),
        .col(col),
        .action(action),
        .next_row(next_row),
        .next_col(next_col),
        .reward(reward),
        .addr_sa(addr_sa),
        .addr_next(addr_next)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;  // 10ns clock period

    // Test procedure
    initial begin
        rst = 1;
        row = 0;
        col = 0;
        action = 0;
        next_row = 0;
        next_col = 0;
        reward = 0;

        // Reset
        #15 rst = 0;

        // Apply some learning updates
        for (int i = 0; i < ROWS; i++) begin
            for (int j = 0; j < COLS; j++) begin
                for (int a = 0; a < ACTIONS; a++) begin
                    row = i;
                    col = j;
                    action = a;
                    next_row = (i + 1) % ROWS;
                    next_col = (j + 1) % COLS;
                    reward = (i + j + a) * 2;

                    #100;  // Wait for Q-update cycle to complete
                end
            end
        end
	print_q_table;
        // Simulation end
        $display("Simulation completed.");
        $finish;
    end
    // Debug task to print Q-table
    task print_q_table;
        $display("\n--- Q-Table Contents ---");
        for (int i = 0; i < ROWS * COLS * ACTIONS; i++) begin
            $display("q_table[%0d] = %0d", i, q_table[i]);
        end
        $display("--- End Q-Table ---\n");
    endtask

endmodule

