// tb_q_learning_update.sv

module tb_q_learning_update;

  // Parameters
  parameter ROWS = 5;
  parameter COLS = 5;
  parameter ACTIONS = 4;
  parameter ADDR_WIDTH = 7;
  parameter DATA_WIDTH = 8;
  parameter QTABLE_SIZE = ROWS * COLS * ACTIONS;

  // Clock and reset
  logic clk, rst;

  // Inputs
  logic [2:0] row, col, next_row, next_col;
  logic [1:0] action;
  logic [7:0] reward;

  // Outputs
  logic [ADDR_WIDTH-1:0] addr_sa;
  logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1];
  logic [DATA_WIDTH-1:0] q_table_out [0:QTABLE_SIZE-1];

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
    .q_table_out(q_table_out)
  );

  // Clock generation
  always #5 clk = ~clk;

  // Initial block
  initial begin
    $display("Starting Q-learning testbench...");
    clk = 0;
    rst = 1;
    row = 3;
    col = 2;
    action = 1;
    next_row = 3;
    next_col = 3;
    reward = 8'd10;

    #10 rst = 0;

    // Run for several cycles
    repeat (20) begin
      @(posedge clk);
    end

    // Display final Q-table values
    $display("\nFinal Q-table:");
    for (int i = 0; i < QTABLE_SIZE; i++) begin
      $display("q_table[%0d] = %0d", i, q_table_out[i]);
    end

    $finish;
  end

endmodule

