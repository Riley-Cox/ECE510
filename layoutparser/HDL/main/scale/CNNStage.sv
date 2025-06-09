module CNNStage #(
    parameter integer IMG_WIDTH = 640,
    parameter integer IMG_HEIGHT = 640,

    parameter integer DATA_WIDTH = 8,
    parameter integer RESULT_WIDTH = 32,

    parameter integer NUM_CHANNELS = 3,
    parameter integer NUM_OUTPUTS = 5,

    parameter bit USE_POOL = 1,
    parameter integer STATE_ID = 0
  
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                             valid_in,
    input  wire signed [NUM_CHANNELS-1:0][DATA_WIDTH-1:0] pixel_in,

    input  wire signed [NUM_OUTPUTS-1:0][NUM_CHANNELS-1:0][2:0][2:0][DATA_WIDTH-1:0] kernel,
    input  wire signed [NUM_OUTPUTS-1:0][RESULT_WIDTH-1:0] bias,

    output logic signed [NUM_OUTPUTS-1:0][RESULT_WIDTH-1:0] pixel_out,
    output logic                             valid_out
);

    logic signed [NUM_CHANNELS-1:0][2:0][2:0][DATA_WIDTH-1:0] window_ch;
    logic [NUM_CHANNELS-1:0] window_valid;
    logic all_valid;
    logic signed [NUM_OUTPUTS-1:0][RESULT_WIDTH-1:0] result_internal;
    logic conv_valid;
    logic signed [NUM_OUTPUTS-1:0][RESULT_WIDTH-1:0] pixel_out_reg;
    logic pixel_valid_reg;

    genvar c;
    generate
        for (c = 0; c < NUM_CHANNELS; c++) begin : win_gen_block
            window_3x3_stream #(
                .DATA_WIDTH(DATA_WIDTH),
                .IMG_WIDTH(IMG_WIDTH),
                .IMG_HEIGHT(IMG_HEIGHT),
                .CHANNEL_ID(c)
            ) win_gen (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .pixel_in(pixel_in[c]),
                .valid_out(window_valid[c]),
                .window(window_ch[c])
            );
        end
    endgenerate


    assign all_valid = &window_valid;




    conv3x3_multi_channel_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .RESULT_WIDTH(RESULT_WIDTH),
        .NUM_INPUT_CHANNELS(NUM_CHANNELS),
        .NUM_OUTPUT_CHANNELS(NUM_OUTPUTS)
    ) conv_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(all_valid),
        .data_in(window_ch),
        .kernel(kernel),
        .bias(bias),
        .result_out(result_internal),
        .valid_out(conv_valid)
    );
    logic [NUM_OUTPUTS-1:0] pool_valid;


    generate
        if (USE_POOL) begin : pool_enabled
            logic signed [NUM_OUTPUTS-1:0][RESULT_WIDTH-1:0] pool_out;

            genvar p;
            for (p = 0; p < NUM_OUTPUTS; p++) begin : pool_block
                maxpool2x2_stream #(
                    .DATA_WIDTH(RESULT_WIDTH),
                    .IMG_WIDTH(IMG_WIDTH - 2),
                    .IMG_HEIGHT(IMG_HEIGHT - 2)
                ) pool_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .valid_in(conv_valid),
                    .pixel_in(result_internal[p]),
                    .valid_out(pool_valid[p]),
                    .pixel_out(pool_out[p])
                );
            end

            assign pixel_out_reg = pool_out;
            assign pixel_valid_reg = &pool_valid;
        end else begin : no_pool
            always_ff @(posedge clk or negedge rst_n) begin
              if (!rst_n) begin
                pixel_out_reg <= '0;
                pixel_valid_reg <= 0;
              end
              else begin
                if (conv_valid) begin
                  pixel_out_reg <= result_internal;
                  pixel_valid_reg <= 1;
                end
                else
                  pixel_valid_reg <= 0;
        end
      end
    end
    endgenerate
  assign pixel_out = pixel_out_reg;
  assign valid_out = pixel_valid_reg;
endmodule

