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

  always_comb begin
  int o,c,i,j;
  for (o = 0; o < NUM_OUTPUTS; o++) begin
    conv_sum[o] = bias[o];
    for (c = 0; c < NUM_CHANNELS; c++) begin
      for (i = 0; i < 3; i++) begin
        for (j = 0; j < 3; j++) begin
          conv_sum[o] += data_in[c][i][j] * kernel[o][c][i][j];
        end
      end
    end
    relu_result[o] = (conv_sum[o] > 0) ? conv_sum[o] : 0;
  end
  end

  always_ff @(posedge clk or negedge rst_n) begin
  int k;
    if (!rst_n) begin
      for (k = 0; k < NUM_OUTPUTS; k++) begin
      result[k]    <= 0;
      end
      valid_out <= 0;
    end else if (valid_in) begin
      for (k = 0; k < NUM_OUTPUTS; k++) begin
      result[k]   <= relu_result[k];
      end
      valid_out <= 1;
    end else begin
      valid_out <= 0;
    end
  end


endmodule

