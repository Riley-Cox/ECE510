module top;

  parameter DATA_WIDTH = 8;
  parameter RESULT_WIDTH = 24;
  parameter IMG_WIDTH = 5;
  parameter IMG_HEIGHT = 5;
  parameter NUM_CHANNELS = 3;
  parameter NUM_OUTPUTS = 4;

  logic clk;
  logic rst_n;
  logic valid_in;
  logic [DATA_WIDTH-1:0] pixel_in_ch [0:NUM_CHANNELS-1];
  logic signed [DATA_WIDTH-1:0] kernel [0:NUM_OUTPUTS-1][0:NUM_CHANNELS-1][0:2][0:2];
  logic signed [RESULT_WIDTH-1:0] bias[0:NUM_OUTPUTS-1];
  logic signed [RESULT_WIDTH-1:0] result[0:NUM_OUTPUTS-1];
  logic signed [RESULT_WIDTH-1:0] expected[0:NUM_OUTPUTS-1][0:(IMG_HEIGHT/2)-1][0:(IMG_WIDTH/2)-1];
  logic signed [RESULT_WIDTH-1:0] conv_map [0:IMG_HEIGHT-3][0:IMG_WIDTH-3];
  logic valid_out;
  int count,i,j,c,o,acc,m,n;

  let max(a,b) = (a > b) ? a : b;

  conv3x3_streaming_top_multi #(
    .DATA_WIDTH(DATA_WIDTH),
    .RESULT_WIDTH(RESULT_WIDTH),
    .IMG_WIDTH(IMG_WIDTH),
    .IMG_HEIGHT(IMG_HEIGHT),
    .NUM_CHANNELS(NUM_CHANNELS),
    .NUM_OUTPUTS(NUM_OUTPUTS)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .pixel_in_ch(pixel_in_ch),
    .kernel(kernel),
    .bias(bias),
    .result(result),
    .valid_out(valid_out)
  );

  logic [DATA_WIDTH-1:0] image [0:NUM_CHANNELS-1][0:IMG_HEIGHT-1][0:IMG_WIDTH-1];

  always
     #5 clk = ~clk;
  

  initial begin
    clk = 0;
    rst_n = 0;
    valid_in = 0;
    for (int c = 0; c < NUM_CHANNELS; c++)
      pixel_in_ch[c] = 0;

    #20 rst_n = 1;

    count = 1;
    for (c = 0; c < NUM_CHANNELS; c++) begin
      for (i = 0; i < IMG_HEIGHT; i++) begin
        for (j = 0; j < IMG_WIDTH; j++) begin
          image[c][i][j] = count;
          count++;
        end
      end
    end
    for (o = 0; o < NUM_OUTPUTS; o++)
    for (c = 0; c < NUM_CHANNELS; c++) begin
      for (i = 0; i < 3; i++) begin
        for (j = 0; j < 3; j++) begin
          kernel[o][c][i][j] = 1;
        end
      end
    end

    for (o = 0; o < NUM_OUTPUTS; o++) bias[o] = 0;

    for (i = 0; i < IMG_HEIGHT; i++) begin
      for (j = 0; j < IMG_WIDTH; j++) begin
        @(posedge clk);
        for (c = 0; c < NUM_CHANNELS; c++) begin
          pixel_in_ch[c] = image[c][i][j];
        end
        valid_in = 1;
      end
    end

    @(posedge clk);
    valid_in = 0;

    repeat (100) @(posedge clk);
    $finish;
  end


  always @(posedge clk) begin
    if (valid_out) begin
      $write("Pooled Results @ (0,0): ");
      for (i = 0; i < NUM_OUTPUTS; i++) begin
        $write("%0d ", result[i]);
        if(result[i] !== expected[i][0][0]) begin
          $fatal(1,"Mismatch at (0,0) output %0d: got %0d, expected %0d", i, result[i], expected[i][0][0]);
        end
      end
      $write("\n");
    end
  end
  
  initial begin
  int b;
    conv_map[0][0] = 1026;
    conv_map[0][1] = 1053;
    conv_map[1][0] = 1134;
    conv_map[1][1] = 1161;
    for (b = 0; b < NUM_OUTPUTS; b++)
      expected[b][0][0] = max(max(conv_map[0][0], conv_map[0][1]), max(conv_map[1][0], conv_map[1][1]));
    end
endmodule

