module conv3x3_streaming_top_multi #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 24,
  parameter IMG_WIDTH = 4,
  parameter IMG_HEIGHT = 4,
  parameter NUM_CHANNELS = 3,
  parameter NUM_OUTPUTS = 4
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  wire [DATA_WIDTH-1:0] pixel_in_ch [0:NUM_CHANNELS-1],
  input  wire signed [DATA_WIDTH-1:0] kernel   [0:NUM_OUTPUTS-1][0:NUM_CHANNELS-1][0:2][0:2],
  input  wire signed [RESULT_WIDTH-1:0] bias [0:NUM_OUTPUTS-1],
  output logic signed [RESULT_WIDTH-1:0] result [0:NUM_OUTPUTS-1],
  output logic valid_out
);

  logic [DATA_WIDTH-1:0] window_ch [0:NUM_CHANNELS-1][0:2][0:2];
  logic [0:NUM_CHANNELS-1] window_valid;
  logic all_valid;
  logic signed [RESULT_WIDTH-1:0] pooled_pixel_out [NUM_OUTPUTS-1:0];
  logic [NUM_OUTPUTS-1:0] pool_valid ;
  logic signed [RESULT_WIDTH-1:0] result_internal [0:NUM_OUTPUTS-1];
  logic signed [RESULT_WIDTH-1:0] result_reg [NUM_OUTPUTS-1:0];
  logic conv_valid, conv_valid_reg;

  int i;


  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      for(i = 0; i < NUM_OUTPUTS; i++) begin
        result_reg[i] <= 0;
      end
        conv_valid_reg <= 0;
    end
    else begin
        conv_valid_reg <= conv_valid;
       if (conv_valid) begin
      for(i = 0; i < NUM_OUTPUTS; i++) begin
        result_reg[i] <= result_internal[i];
      end
    end
  end 
  end

  genvar c;
  generate
    for (c = 0; c < NUM_CHANNELS; c++) begin : win_gen_block
      window_3x3_stream #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
      ) win_gen (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .pixel_in(pixel_in_ch[c]),
        .valid_out(window_valid[c]),
        .window(window_ch[c])
      );
    end
  endgenerate

  genvar p;
  generate
    for (p = 0; p < NUM_OUTPUTS; p++) begin : pool_block
      maxpool2x2_stream #(
        .DATA_WIDTH(RESULT_WIDTH),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
      ) pool_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(conv_valid_reg),
        .pixel_in(result_reg[p]),
        .valid_out(pool_valid[p]),
        .pixel_out(pooled_pixel_out[p])
      );
      assign result[p] = pooled_pixel_out[p];
    end
  endgenerate

  assign valid_out  = |pool_valid;
  assign all_valid = &window_valid;

  conv3x3_multi_channel_core #(
    .DATA_WIDTH(DATA_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .NUM_CHANNELS(NUM_CHANNELS),
    .NUM_OUTPUTS(NUM_OUTPUTS)
  ) conv_core (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(all_valid),
    .data_in(window_ch),
    .kernel(kernel),
    .bias(bias),
    .result(result_internal),
    .valid_out(conv_valid)
  );


endmodule
