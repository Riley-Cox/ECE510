module conv3x3_core #(
  parameter DATA_WIDTH = 8,
  parameter RESULT_WIDTH = 20
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  wire signed [DATA_WIDTH-1:0] data_in [0:2][0:2],
  input  wire signed [DATA_WIDTH-1:0] kernel  [0:2][0:2],
  input  logic signed [RESULT_WIDTH-1:0] bias,
  output logic signed [RESULT_WIDTH-1:0] result,
  output logic valid_out
);

  logic signed [RESULT_WIDTH-1:0] mac_sum;

  always_comb begin
    mac_sum = 0;
    for (int i = 0; i < 3; i++) begin
      for (int j = 0; j < 3; j++) begin
        mac_sum += data_in[i][j] * kernel[i][j];
      end
    end
    mac_sum = mac_sum + bias;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      result    <= 0;
      valid_out <= 0;
    end else if (valid_in) begin
      result    <= mac_sum;
      valid_out <= 1;
    end else begin
      valid_out <= 0;
    end
  end

endmodule
