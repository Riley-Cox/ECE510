module top;

  parameter DATA_WIDTH = 8;
  parameter RESULT_WIDTH = 24;
  parameter IMG_WIDTH = 22;
  parameter IMG_HEIGHT = 22;
  parameter NUM_INPUT_CHANNELS = 3;
  parameter NUM_MID_CHANNELS = 4;
  parameter NUM_OUTPUT_CHANNELS = 4;
  parameter NUM_FINAL_CHANNELS = 2;

  logic clk;
  logic rst_n;
  logic valid_in;
  logic [DATA_WIDTH-1:0] pixel_in_ch [0:NUM_INPUT_CHANNELS-1];

  logic signed [DATA_WIDTH-1:0] kernel1 [0:NUM_MID_CHANNELS-1][0:NUM_INPUT_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias1 [0:NUM_MID_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] kernel2 [0:NUM_OUTPUT_CHANNELS-1][0:NUM_MID_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias2 [0:NUM_OUTPUT_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] kernel3 [0:NUM_FINAL_CHANNELS-1][0:NUM_OUTPUT_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias3 [0:NUM_FINAL_CHANNELS-1];

  logic signed [RESULT_WIDTH-1:0] result [0:NUM_FINAL_CHANNELS-1];
  logic valid_out;

  int i, j, c, m, o, f, val;
  logic [DATA_WIDTH-1:0] image [0:NUM_INPUT_CHANNELS-1][0:IMG_HEIGHT-1][0:IMG_WIDTH-1];

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

  always #5 clk = ~clk;

  int result_fd;

  initial begin
    result_fd = $fopen("results.txt", "w");
    if (result_fd == 0) begin
      $display("ERROR: Could not open results.txt for writing.");
      $finish;
    end
  end


  initial begin
    clk = 0;
    rst_n = 0;
    valid_in = 0;
    for (c = 0; c < NUM_INPUT_CHANNELS; c++) pixel_in_ch[c] = 0;

    #20 rst_n = 1;

    val = 1;
    for (c = 0; c < NUM_INPUT_CHANNELS; c++) begin
      for (i = 0; i < IMG_HEIGHT; i++) begin
        for (j = 0; j < IMG_WIDTH; j++) begin
          image[c][i][j] = val;
          val++;
        end
      end
    end

    for (m = 0; m < NUM_MID_CHANNELS; m++) begin
      for (c = 0; c < NUM_INPUT_CHANNELS; c++) begin
        for (i = 0; i < 3; i++) begin
          for (j = 0; j < 3; j++) begin
            kernel1[m][c][i][j] = 1;
          end
        end
      end
      bias1[m] = 0;
    end

    for (o = 0; o < NUM_OUTPUT_CHANNELS; o++) begin
      for (m = 0; m < NUM_MID_CHANNELS; m++) begin
        for (i = 0; i < 3; i++) begin
          for (j = 0; j < 3; j++) begin
            kernel2[o][m][i][j] = 1;
          end
        end
      end
      bias2[o] = 0;
    end

    for (f = 0; f < NUM_FINAL_CHANNELS; f++) begin
      for (o = 0; o < NUM_OUTPUT_CHANNELS; o++) begin
        for (i = 0; i < 3; i++) begin
          for (j = 0; j < 3; j++) begin
            kernel3[f][o][i][j] = 1;
          end
        end
      end
      bias3[f] = 0;
    end

    for (i = 0; i < IMG_HEIGHT; i++) begin
      for (j = 0; j < IMG_WIDTH; j++) begin
        @(posedge clk);
        for (c = 0; c < NUM_INPUT_CHANNELS; c++) begin
          pixel_in_ch[c] = image[c][i][j];
        end
        valid_in = 1;
      end
    end

    @(posedge clk);
    valid_in = 0;
    repeat (10000) @(posedge clk);
    $finish;
  end

  always @(posedge clk) begin
    if (valid_out) begin
      $fwrite(result_fd, "Final Output: ");
      for (int f = 0; f < NUM_FINAL_CHANNELS; f++) begin
        $fwrite(result_fd, "%0d ", result[f]);
      end
      $fwrite(result_fd, "\n");
    end
  end

  final begin
    $fclose(result_fd);
  end
endmodule

