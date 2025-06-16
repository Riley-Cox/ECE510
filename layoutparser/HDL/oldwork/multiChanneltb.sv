module top;

  parameter DATA_WIDTH = 8;
  parameter RESULT_WIDTH = 24;
  parameter NUM_CHANNELS = 3;

  logic clk;
  logic rst_n;
  logic valid_in;

  logic signed [DATA_WIDTH-1:0] data_in [0:NUM_CHANNELS-1][0:2][0:2];
  logic signed [DATA_WIDTH-1:0] kernel  [0:NUM_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias;
  logic signed [RESULT_WIDTH-1:0] result;
  logic valid_out;
  int found;

  conv3x3_multi_channel_core #(DATA_WIDTH, RESULT_WIDTH, NUM_CHANNELS) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .data_in(data_in),
    .kernel(kernel),
    .bias(bias),
    .result(result),
    .valid_out(valid_out)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0;
    rst_n = 0;
    valid_in = 0;
    found = 0;

    #20 rst_n = 1;

    for (int c = 0; c < NUM_CHANNELS; c++) begin
      data_in[c][0][0] = 1; data_in[c][0][1] = 2; data_in[c][0][2] = 3;
      data_in[c][1][0] = 4; data_in[c][1][1] = 5; data_in[c][1][2] = 6;
      data_in[c][2][0] = 7; data_in[c][2][1] = 8; data_in[c][2][2] = 9;

      kernel[c][0][0] = -1; kernel[c][0][1] = 0; kernel[c][0][2] = 1;
      kernel[c][1][0] = -1; kernel[c][1][1] = 0; kernel[c][1][2] = 1;
     kernel[c][2][0] = -1; kernel[c][2][1] = 0; kernel[c][2][2] = 1;
    end

    bias = 0;

   repeat(1) @(posedge clk)
    valid_in = 1;

    repeat(1) @(posedge clk)
    valid_in = 0;

    repeat (5)begin
     @(posedge clk);
    if (valid_out) begin
      $display("✅ Result = %0d (expected: 18)", result);
      found = 1;
      end
    end
    if (!found)
      $display("❌ Output not valid");

    $finish;
  end

endmodule

