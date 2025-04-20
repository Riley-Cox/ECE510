// tb_q_learning_update.sv

`timescale 1ns/1ps

module tb_q_learning_update;

    // Parameters
    parameter ROWS = 5;
    parameter COLS = 5;
    parameter ACTIONS = 4;
    parameter ADDR_WIDTH = 7;
    parameter DATA_WIDTH = 8;
    parameter TABLE_SIZE = ROWS * COLS * ACTIONS;

    // Testbench signals
    logic clk;
    logic rst;
    logic [2:0] row, col, next_row, next_col;
    logic [1:0] action;
    logic [7:0] reward;
    logic [ADDR_WIDTH-1:0] addr_sa;
    logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1];
    logic [DATA_WIDTH-1:0] q_table [0:TABLE_SIZE-1];

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

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
        .addr_next(addr_next),
        .q_table(q_table)
    );

    // Stimulus
    initial begin
        // Initialize
        rst = 1;
        row = 0;
        col = 0;
        action = 0;
        next_row = 0;
        next_col = 1;
        reward = 8'd10;

        // Clear Q-table
        for (int i = 0; i < TABLE_SIZE; i++) begin
            q_table[i] = 0;
        end

        // Apply reset
        #10;
        rst = 0;

        // Wait for a few clock cycles
        #100;

        // Display results
        $display("Q[%0d] = %0d", addr_sa, q_table[addr_sa]);

        #100;
        $finish;
    end

endmodule

