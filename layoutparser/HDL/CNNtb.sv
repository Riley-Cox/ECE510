module tb_conv3x3_core;

    parameter DATA_WIDTH = 8;
    parameter RESULT_WIDTH = 20;

    logic clk;
    logic rst_n;
    logic valid_in;
    logic signed [DATA_WIDTH-1:0] data_in [0:2][0:2];
    logic signed [DATA_WIDTH-1:0] kernel  [0:2][0:2];
    logic signed [RESULT_WIDTH-1:0] bias;
    logic signed [RESULT_WIDTH-1:0] result;
    logic valid_out;

    conv3x3_core #(DATA_WIDTH, RESULT_WIDTH) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in),
        .kernel(kernel),
        .bias(bias),
        .result(result),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk; // 10 ns period

    initial begin
        $display("Running 3x3 Convolution Test...");
        clk = 0;
        rst_n = 0;
        valid_in = 0;

        #20 rst_n = 1;

        // Input patch
        data_in[0][0] = 1;  data_in[0][1] = 2;  data_in[0][2] = 3;
        data_in[1][0] = 4;  data_in[1][1] = 5;  data_in[1][2] = 6;
        data_in[2][0] = 7;  data_in[2][1] = 8;  data_in[2][2] = 9;

        // Kernel
        kernel[0][0] = -1; kernel[0][1] = 0; kernel[0][2] = 1;
        kernel[1][0] = -1; kernel[1][1] = 0; kernel[1][2] = 1;
        kernel[2][0] = -1; kernel[2][1] = 0; kernel[2][2] = 1;

        bias = 0;

        #10 valid_in = 1;
        #10 valid_in = 0;

        #10;
        if (valid_out)
            $display("Result = %0d", result);
        else
            $display("Result not valid");

        #20 $finish;
    end

endmodule

