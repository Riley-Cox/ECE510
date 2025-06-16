module dual_stage_wrapper #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 24,
  parameter IMG_WIDTH = 20,
  parameter IMG_HEIGHT = 20,
  parameter NUM_INPUT_CHANNELS = 3,
  parameter NUM_MID_CHANNELS = 4,
  parameter NUM_OUTPUT_CHANNELS = 4,
  parameter NUM_FINAL_CHANNELS = 4
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
  logic signed [RESULT_WIDTH-1:0] stage1_out_reg [0:NUM_MID_CHANNELS-1];
  logic signed [RESULT_WIDTH-1:0] stage2_out_reg [0:NUM_OUTPUT_CHANNELS-1];
  logic stage1_valid;
  logic stage1_valid_reg;
  logic stage2_valid_reg;
  logic signed [RESULT_WIDTH-1:0] stage2_out [0:NUM_OUTPUT_CHANNELS-1];
  logic stage2_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage1_valid_reg <= 0;
      for (int i = 0; i < NUM_MID_CHANNELS; i++)
        stage1_out_reg[i] <= 0;
    end
    else begin
      stage1_valid_reg <= stage1_valid;
      for (int i = 0; i < NUM_MID_CHANNELS; i++)
        stage1_out_reg[i] <= stage1_out[i];
      end
    end
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage2_valid_reg <= 0;
      for (int i = 0; i < NUM_OUTPUT_CHANNELS; i++)
        stage2_out_reg[i] <= 0;
    end
    else begin
      stage2_valid_reg <= stage2_valid;
      for (int i = 0; i < NUM_OUTPUT_CHANNELS; i++)
        stage2_out_reg[i] <= stage2_out[i];
      end
    end


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
    .IMG_WIDTH(9),
    .IMG_HEIGHT(9),
    .NUM_CHANNELS(NUM_MID_CHANNELS),
    .NUM_OUTPUTS(NUM_OUTPUT_CHANNELS)
  ) stage2 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(stage1_valid_reg),
    .pixel_in_ch(stage1_out_reg),
    .kernel(kernel2),
    .bias(bias2),
    .result(stage2_out),
    .valid_out(stage2_valid)
  );

  conv3x3_streaming_top_multi #(
    .DATA_WIDTH(RESULT_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(3),
    .IMG_HEIGHT(3),
    .NUM_CHANNELS(NUM_OUTPUT_CHANNELS),
    .NUM_OUTPUTS(NUM_FINAL_CHANNELS)
  ) stage3 (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(stage2_valid_reg),
    .pixel_in_ch(stage2_out_reg),
    .kernel(kernel3),
    .bias(bias3),
    .result(result),
    .valid_out(valid_out)
  );
  int i;
  initial i = 0;
  always @(posedge clk)
    if (stage3.valid_out == 1) begin
      $display("counter: %d time: %0d",i, $time);
      i++;
      end
  always @(posedge clk)
    if(stage3.valid_in)
      $display("Stage3_in: %0d %0d %0d %0d time: %0d",stage3.pixel_in_ch[0], stage3.pixel_in_ch[1], stage3.pixel_in_ch[2], stage3.pixel_in_ch[3], $time);
  always @(posedge clk)
    if(stage3.valid_out)
      $display("Result: %0d %0d %0d %0d",result[0],result[1], result[2], result[3]);
 
  always @(posedge clk)
    if (stage3.valid_out) begin
      if (^stage3.result[0] === 1'bx)
        $display(" X Stage 3 result[0] is X at time %0t", $time);
      else
        $display(" Stage3 result[0] = %0d at time %0t", stage3.result[0], $time); 
      end

  
endmodule

