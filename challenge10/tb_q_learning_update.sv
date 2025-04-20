module tb_q_learning_update;

    // Declare the necessary signals
    logic clk;
    logic rst;
    logic [2:0] row, col, next_row, next_col;
    logic [1:0] action;
    logic [7:0] reward;
    logic [ADDR_WIDTH-1:0] addr_sa;
    logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1];
    logic [DATA_WIDTH-1:0] q_table_out [0:(ROWS*COLS*ACTIONS)-1];  // New output to observe q_table

    // Instantiate the q_learning_update module
    q_learning_update #(
        .ROWS(5),
        .COLS(5),
        .ACTIONS(4),
        .ADDR_WIDTH(7),
        .DATA_WIDTH(8)
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
        .q_table_out(q_table_out)  // Connecting the output
    );

    // Clock generation
    always #5 clk = ~clk;

    // Stimulus block
    initial begin
        // Initialize the signals
        clk = 0;
        rst = 1;
        row = 0;
        col = 0;
        action = 0;
        next_row = 1;
        next_col = 1;
        reward = 10;

        // Apply reset
        #10 rst = 0;
        #10 rst = 1;
        #10 rst = 0;

        // Apply stimulus
        row = 1; col = 1; action = 2; next_row = 2; next_col = 2; reward = 5;

        // Add more test cases as needed

        // Monitor the q_table_out at each cycle
        #10;
        $display("q_table[0]: %h", q_table_out[0]);
        $display("q_table[1]: %h", q_table_out[1]);
        // Print more entries if needed

        // End the simulation
        #100 $finish;
    end
endmodule

