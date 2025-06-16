module conv3x3_multi_channel_core #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 24,
  parameter NUM_CHANNELS = 3,
  parameter NUM_OUTPUTS = 4
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  wire signed [DATA_WIDTH-1:0] data_in  [0:NUM_CHANNELS-1][0:2][0:2],
  input  wire signed [DATA_WIDTH-1:0] kernel   [0:NUM_OUTPUTS-1][0:NUM_CHANNELS-1][0:2][0:2],
  input  wire signed [RESULT_WIDTH-1:0] bias [0:NUM_OUTPUTS-1],
  output logic signed [RESULT_WIDTH-1:0] result [0:NUM_OUTPUTS-1],
  output logic valid_out
);

  logic signed [RESULT_WIDTH-1:0] conv_sum [0:NUM_OUTPUTS-1];
  logic signed [RESULT_WIDTH-1:0] relu_result [0:NUM_OUTPUTS-1];

  logic valid_in_d;
int i;
always @(posedge clk)
  if(valid_out)
    $display("CNN output: %0d %0d %0d %0d time: %0d",result[0],result[1],result[2],result[3], $time);


always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n)
    valid_in_d <= 0;
  else
    valid_in_d <= valid_in;
  end

  logic signed [RESULT_WIDTH-1:0] acc;

  always_comb begin
  int o,c,i,j;
  acc = 0;
  for (o = 0; o < NUM_OUTPUTS; o++) begin
    if (valid_in_d) begin
    acc = bias[o];
    for (c = 0; c < NUM_CHANNELS; c++) begin
      for (i = 0; i < 3; i++) begin
        for (j = 0; j < 3; j++) begin
          acc += data_in[c][i][j] * kernel[o][c][i][j];
        end
      end
    end
    relu_result[o] = (acc > 0) ? acc : 0;
  end
  end
  end

  always_ff @(posedge clk or negedge rst_n) begin
  int k;
    if (!rst_n) begin
      for (k = 0; k < NUM_OUTPUTS; k++) begin
      result[k]    <= 0;
      end
      valid_out <= 0;
    end else if (valid_in_d) begin
      for (k = 0; k < NUM_OUTPUTS; k++) begin
      result[k]   <= relu_result[k];
      end
    end else begin
      valid_out <= 0;
    end
      valid_out <= valid_in_d;
  end


endmodule

