module conv3x3_multi_channel_core #(
    parameter integer NUM_INPUT_CHANNELS  = 3,
    parameter integer NUM_OUTPUT_CHANNELS = 2,
    parameter integer DATA_WIDTH = 8,
    parameter integer RESULT_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,

    input  wire signed [NUM_INPUT_CHANNELS-1:0][2:0][2:0][DATA_WIDTH-1:0] data_in,
    input  wire signed [NUM_OUTPUT_CHANNELS-1:0][NUM_INPUT_CHANNELS-1:0][2:0][2:0][DATA_WIDTH-1:0] kernel,
    input  wire signed [NUM_OUTPUT_CHANNELS-1:0][RESULT_WIDTH-1:0] bias,

    output logic signed [NUM_OUTPUT_CHANNELS-1:0][RESULT_WIDTH-1:0] result_out,
    output logic valid_out
);

    logic signed [63:0] acc [NUM_OUTPUT_CHANNELS-1:0];
    logic signed [NUM_OUTPUT_CHANNELS-1:0][RESULT_WIDTH-1:0] relu_result;

    always_comb begin
        for (int o = 0; o < NUM_OUTPUT_CHANNELS; o++) begin
            acc[o] = bias[o];
            for (int c = 0; c < NUM_INPUT_CHANNELS; c++) begin
                for (int i = 0; i < 3; i++) begin
                    for (int j = 0; j < 3; j++) begin
                        acc[o] += (data_in[c][i][j] * kernel[o][c][i][j]);
                    end
                end
            end
            relu_result[o] = (acc[o] > 0) ? acc[o] : 0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int k = 0; k < NUM_OUTPUT_CHANNELS; k++) begin
                result_out[k] <= '0;
            end
            valid_out <= 0;
        end else if (valid_in) begin
            for (int k = 0; k < NUM_OUTPUT_CHANNELS; k++) begin
                result_out[k] <= relu_result[k];
            end
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule

