module top;

  parameter DATA_WIDTH = 8;
  parameter RESULT_WIDTH = 24;
  parameter IMG_WIDTH = 20;
  parameter IMG_HEIGHT = 20;
  parameter NUM_INPUT_CHANNELS = 3;
  parameter NUM_MID_CHANNELS = 4;
  parameter NUM_OUTPUT_CHANNELS = 4;
  parameter NUM_FINAL_CHANNELS = 4;

  logic clk;
  logic rst_n;
  logic valid_in;
  logic [DATA_WIDTH-1:0] pixel_in_ch[0:NUM_INPUT_CHANNELS-1];

  logic signed [DATA_WIDTH-1:0] kernel1[0:NUM_MID_CHANNELS-1][0:NUM_INPUT_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias1[0:NUM_MID_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] kernel2[0:NUM_OUTPUT_CHANNELS-1][0:NUM_MID_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias2[0:NUM_OUTPUT_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] kernel3[0:NUM_FINAL_CHANNELS-1][0:NUM_OUTPUT_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias3[0:NUM_FINAL_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] result[0:NUM_FINAL_CHANNELS-1];
  logic signed [RESULT_WIDTH-1:0] result_d[0:NUM_FINAL_CHANNELS-1];
  logic valid_out, valid_out_d;

  logic [DATA_WIDTH-1:0] image[0:NUM_INPUT_CHANNELS-1][0:IMG_HEIGHT-1][0:IMG_WIDTH-1];

  int i, j, c, f;

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    valid_out_d <= valid_out;
    result_d <= result;
  end

  dual_stage_wrapper #(
    .DATA_WIDTH(DATA_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(IMG_WIDTH),
    .IMG_HEIGHT(IMG_HEIGHT),
    .NUM_INPUT_CHANNELS(NUM_INPUT_CHANNELS),
    .NUM_MID_CHANNELS(NUM_MID_CHANNELS),
    .NUM_OUTPUT_CHANNELS(NUM_OUTPUT_CHANNELS),
    .NUM_FINAL_CHANNELS(NUM_FINAL_CHANNELS)
     ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .pixel_in_ch(pixel_in_ch),
    .kernel1(kernel1),
    .bias1(bias1),
    .kernel2(kernel2),
    .bias2(bias2),
    .kernel3(kernel3),
    .bias3(bias3),
    .result(result),
    .valid_out(valid_out)
  );

  initial begin
    clk = 0;
    rst_n = 0;
    valid_in = 0;
    for (c = 0; c < NUM_INPUT_CHANNELS; c++) pixel_in_ch[c] = 0;

    $readmemh("image_input.mem", image);
    $readmemh("kernel1.mem", kernel1);
    $readmemh("bias1.mem", bias1);
    $readmemh("kernel2.mem", kernel2);
    $readmemh("bias2.mem", bias2);
    $readmemh("kernel3.mem", kernel3);
    $readmemh("bias3.mem", bias3);

    #20 rst_n = 1;
    repeat(5) @(posedge clk);
    #20 rst_n = 0;
    #20 rst_n = 1; 
    #20

    for (i = 0; i < IMG_HEIGHT; i++) begin
      for (j = 0; j < IMG_WIDTH; j++) begin
        for (c = 0; c < NUM_INPUT_CHANNELS; c++)
          pixel_in_ch[c] = image[c][i][j];
          valid_in = 1;
        @(posedge clk);
      end
    end
   repeat (20) @(posedge clk);
    valid_in = 0;

    repeat (20000) @(posedge clk);

    $fclose(f);
    $finish;
  end

  initial begin
    $dumpfile("cnn_testbench.vcd");
    $dumpvars(0, top);
  end

  initial begin
    
    f = $fopen("results.txt", "w");
    forever begin
      @(posedge clk);
      if (valid_out_d) begin
        $display("Writing to file");
        $fwrite(f, "Output: ");
        for (int o = 0; o < NUM_FINAL_CHANNELS; o++) begin
          if(result_d[o] !== 1'bx)
          $fwrite(f, "%0d ", result_d[o]);
        end
        $fwrite(f, "\n");
      end
    end
  end


endmodule

