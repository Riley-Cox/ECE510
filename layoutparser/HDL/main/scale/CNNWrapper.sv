module CNNWrapper #(
    parameter integer IMG_WIDTH     = 640,
    parameter integer IMG_HEIGHT    = 640,

    parameter integer CH0_IN  = 3,
    parameter integer CH0_OUT = 16,
    parameter integer CH1_OUT = 32,
    parameter integer CH2_OUT = 64,
    parameter integer CH3_OUT = 64,
    parameter integer CH4_OUT = 64,
    parameter integer CH5_OUT = 2,

    parameter integer DATA_WIDTH   = 8,
    parameter integer RESULT_WIDTH = 32
)(
    input  logic clk,
    input  logic rst_n,

    input  logic valid_in,
    input  wire signed [CH0_IN-1:0][DATA_WIDTH-1:0] image_in,

    input  wire signed [CH0_OUT-1:0][CH0_IN-1:0][2:0][2:0][DATA_WIDTH-1:0] kernel0,
    input  wire signed [CH0_OUT-1:0][RESULT_WIDTH-1:0] bias0,

    input  wire signed [CH1_OUT-1:0][CH0_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel1,
    input  wire signed [CH1_OUT-1:0][RESULT_WIDTH-1:0] bias1,

    input  wire signed [CH2_OUT-1:0][CH1_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel2,
    input  wire signed [CH2_OUT-1:0][RESULT_WIDTH-1:0] bias2,

    input  wire signed [CH3_OUT-1:0][CH2_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel3,
    input  wire signed [CH3_OUT-1:0][RESULT_WIDTH-1:0] bias3,

    input  wire signed [CH4_OUT-1:0][CH3_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel4,
    input  wire signed [CH4_OUT-1:0][RESULT_WIDTH-1:0] bias4,

    input  wire signed [CH5_OUT-1:0][CH4_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel5,
    input  wire signed [CH5_OUT-1:0][RESULT_WIDTH-1:0] bias5,

    output logic signed [CH5_OUT-1:0][RESULT_WIDTH-1:0] final_out,
    output logic                                       valid_out
);

    logic signed [CH0_OUT-1:0][RESULT_WIDTH-1:0] stage0_out;
    logic stage0_valid;

    CNNStage #(
        .IMG_WIDTH(640), .IMG_HEIGHT(640),
        .DATA_WIDTH(DATA_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH0_IN), .NUM_OUTPUTS(CH0_OUT),
        .USE_POOL(0),
        .STATE_ID(0)
    ) stage0 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .pixel_in(image_in),
        .kernel(kernel0), .bias(bias0),
        .pixel_out(stage0_out),
        .valid_out(stage0_valid)
    );
    logic signed [CH1_OUT-1:0][RESULT_WIDTH-1:0] stage1_out;
    logic stage1_valid;

    CNNStage #(
        .IMG_WIDTH(319), .IMG_HEIGHT(319),
        .DATA_WIDTH(RESULT_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH0_OUT), .NUM_OUTPUTS(CH1_OUT),
        .USE_POOL(0),
        .STATE_ID(1)
    ) stage1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(stage0_valid),
        .pixel_in(stage0_out),
        .kernel(kernel1), .bias(bias1),
        .pixel_out(stage1_out),
        .valid_out(stage1_valid)
    );

    logic signed [CH2_OUT-1:0][RESULT_WIDTH-1:0] stage2_out;
    logic stage2_valid;

    CNNStage #(
        .IMG_WIDTH(158), .IMG_HEIGHT(158),
        .DATA_WIDTH(RESULT_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH1_OUT), .NUM_OUTPUTS(CH2_OUT),
        .USE_POOL(0),
        .STATE_ID(2)
    ) stage2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(stage1_valid),
        .pixel_in(stage1_out),
        .kernel(kernel2), .bias(bias2),
        .pixel_out(stage2_out),
        .valid_out(stage2_valid)
    );

    logic signed [CH3_OUT-1:0][RESULT_WIDTH-1:0] stage3_out;
    logic stage3_valid;

    CNNStage #(
        .IMG_WIDTH(78), .IMG_HEIGHT(78),
        .DATA_WIDTH(RESULT_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH2_OUT), .NUM_OUTPUTS(CH3_OUT),
        .USE_POOL(0),
        .STATE_ID(3)
    ) stage3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(stage2_valid),
        .pixel_in(stage2_out),
        .kernel(kernel3), .bias(bias3),
        .pixel_out(stage3_out),
        .valid_out(stage3_valid)
    );

    logic signed [CH4_OUT-1:0][RESULT_WIDTH-1:0] stage4_out;
    logic stage4_valid;

    CNNStage #(
        .IMG_WIDTH(38), .IMG_HEIGHT(38),
        .DATA_WIDTH(RESULT_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH3_OUT), .NUM_OUTPUTS(CH4_OUT),
        .USE_POOL(0),
        .STATE_ID(4)
    ) stage4 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(stage3_valid),
        .pixel_in(stage3_out),
        .kernel(kernel4), .bias(bias4),
        .pixel_out(stage4_out),
        .valid_out(stage4_valid)
    );

    CNNStage #(
        .IMG_WIDTH(18), .IMG_HEIGHT(18),
        .DATA_WIDTH(RESULT_WIDTH), .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_CHANNELS(CH4_OUT), .NUM_OUTPUTS(CH5_OUT),
        .USE_POOL(0),
        .STATE_ID(5)
    ) stage5 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(stage4_valid),
        .pixel_in(stage4_out),
        .kernel(kernel5), .bias(bias5),
        .pixel_out(final_out),
        .valid_out(valid_out)
    );

endmodule

