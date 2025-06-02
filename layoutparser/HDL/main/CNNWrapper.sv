module dual_stage_wrapper #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 24,
  parameter IMG_WIDTH = 10,
  parameter IMG_HEIGHT = 10,
  parameter NUM_INPUT_CHANNELS = 3,
  parameter NUM_MID_CHANNELS = 4,
  parameter NUM_OUTPUT_CHANNELS = 4,
  parameter NUM_FINAL_CHANNELS = 2
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  wire [DATA_WIDTH-1:0] pixel_in_ch [0:NUM_INPUT_CHANNELS-1],

  input  wire signed [DATA_WIDTH-1:0] kernel1 [0:NUM_MID_CHANNELS-1][0:NUM_INPUT_CHANNELS-1][0:2][0:2],
  input  wire signed [RESULT_WIDTH-1:0] bias1 [0:NUM_MID_CHANNELS-1],

  input  wire signed [RESULT_WIDTH-1:0] kernel2 [0:NUM_OUTPUT_CHANNELS-1][0:NUM_MID_CHANNELS-1][0:2][0:2],
  input  wire signed [RESULT_WIDTH-1:0] bias2 [0:NUM_OUTPUT_CHANNELS-1],

  input  wire signed [RESULT_WIDTH-1:0] kernel3 [0:NUM_FINAL_CHANNELS-1][0:NUM_OUTPUT_CHANNELS-1][0:2][0:2],
  input  wire signed [RESULT_WIDTH-1:0] bias3 [0:NUM_FINAL_CHANNELS-1],

  output logic signed [RESULT_WIDTH-1:0] result [0:NUM_FINAL_CHANNELS-1],
  output logic valid_out
);

  logic signed [RESULT_WIDTH-1:0] stage1_out [0:NUM_MID_CHANNELS-1];
  logic stage1_valid;

  logic signed [RESULT_WIDTH-1:0] stage2_out [0:NUM_OUTPUT_CHANNELS-1];
  logic stage2_valid;

  conv3x3_streaming_top_multi #(
    .DATA_WIDTH(DATA_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(IMG_WIDTH),
    .IMG_HEIGHT(IMG_HEIGHT),
    .NUM_CHANNELS(NUM_INPUT_CHANNELS),
    .NUM_OUTPUTS(NUM_MID_CHANNELS)
  ) stage1 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .pixel_in_ch(pixel_in_ch),
    .kernel(kernel1),
    .bias(bias1),
    .result(stage1_out),
    .valid_out(stage1_valid)
  );

  conv3x3_streaming_top_multi #(
    .DATA_WIDTH(RESULT_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(IMG_WIDTH/2),
    .IMG_HEIGHT(IMG_HEIGHT/2),
    .NUM_CHANNELS(NUM_MID_CHANNELS),
    .NUM_OUTPUTS(NUM_OUTPUT_CHANNELS)
  ) stage2 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(stage1_valid),
    .pixel_in_ch(stage1_out),
    .kernel(kernel2),
    .bias(bias2),
    .result(stage2_out),
    .valid_out(stage2_valid)
  );

  conv3x3_streaming_top_multi #(
    .DATA_WIDTH(RESULT_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(IMG_WIDTH/4),
    .IMG_HEIGHT(IMG_HEIGHT/4),
    .NUM_CHANNELS(NUM_OUTPUT_CHANNELS),
    .NUM_OUTPUTS(NUM_FINAL_CHANNELS)
  ) stage3 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(stage2_valid),
    .pixel_in_ch(stage2_out),
    .kernel(kernel3),
    .bias(bias3),
    .result(result),
    .valid_out(valid_out)
  );
  always @(posedge clk)
    if (valid_out) 
      $display("Result is; %p", result);
  always @(posedge clk)
    if (stage2_valid)
      $display("stage2_out: %0d %0d %0d %0d", stage2_out[0], stage2_out[1], stage2_out[2], stage2_out[3]);
endmodule

