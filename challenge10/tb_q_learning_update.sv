// tb_q_learning_update.sv

module tb_q_learning_update;
    // Parameters
    parameter ROWS = 5;
    parameter COLS = 5;
    parameter ACTIONS = 4;
    parameter ADDR_WIDTH = 7;
    parameter DATA_WIDTH = 8;

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

    // Instantiate DUT
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
    always #5 clk = ~clk;

    // Stimulus
    initial begin
        clk = 0;
        rst = 1;
        row = 3'd0;
        col = 3'd0;
        action = 2'd0;
        next_row = 3'd0;
        next_col = 3'd1;
        reward = 8'd10;

        #10 rst = 0;

        repeat (10) begin
            @(posedge clk);
        end

        $display("\nFinal Q-table entries:");
        for (int i = 0; i < ROWS * COLS * ACTIONS; i++) begin
            $display("q_table[%0d] = %0d", i, dut.q_table[i]);
        end

        $finish;
    end
endmodule

