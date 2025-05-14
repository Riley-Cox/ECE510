module conv3x3_multi_channel_core #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 24,
  parameter NUM_CHANNELS = 3
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  wire signed [DATA_WIDTH-1:0] data_in  [0:NUM_CHANNELS-1][0:2][0:2],
  input  wire signed [DATA_WIDTH-1:0] kernel   [0:NUM_CHANNELS-1][0:2][0:2],
  input  logic signed [RESULT_WIDTH-1:0] bias,
  output logic signed [RESULT_WIDTH-1:0] result,
  output logic valid_out
);

  logic signed [RESULT_WIDTH-1:0] conv_sum;
  logic signed [RESULT_WIDTH-1:0] relu_result;

  always_comb begin
    conv_sum = 0;
    for (int c = 0; c < NUM_CHANNELS; c++) begin
      for (int i = 0; i < 3; i++) begin
        for (int j = 0; j < 3; j++) begin
          conv_sum += data_in[c][i][j] * kernel[c][i][j];
        end
      end
    end
    conv_sum = conv_sum + bias;
    relu_result = (conv_sum > 0) ? conv_sum : 0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      result    <= 0;
      valid_out <= 0;
    end else if (valid_in) begin
      result    <= relu_result;
      valid_out <= 1;
    end else begin
      valid_out <= 0;
    end
  end

endmodule

