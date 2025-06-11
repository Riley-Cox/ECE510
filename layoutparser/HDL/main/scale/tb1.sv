`timescale 1ns/1ps

module top;

  parameter IMG_WIDTH = 640;
  parameter IMG_HEIGHT = 640;

  parameter CH0_IN  = 3;
  parameter CH0_OUT = 16;
  parameter CH1_OUT = 32;
  parameter CH2_OUT = 64;
  parameter CH3_OUT = 64;
  parameter CH4_OUT = 64;
  parameter CH5_OUT = 2;
  parameter CLK_PERIOD_NS = 10;

  parameter DATA_WIDTH   = 8;
  parameter RESULT_WIDTH = 32;

  logic clk = 0;
  logic rst_n;
  logic valid_in;
  logic signed [CH0_IN-1:0][DATA_WIDTH-1:0] image_in;

  logic signed [CH0_OUT-1:0][CH0_IN-1:0][2:0][2:0][DATA_WIDTH-1:0] kernel0;
  logic signed [CH1_OUT-1:0][CH0_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel1;
  logic signed [CH2_OUT-1:0][CH1_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel2;
  logic signed [CH3_OUT-1:0][CH2_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel3;
  logic signed [CH4_OUT-1:0][CH3_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel4;
  logic signed [CH5_OUT-1:0][CH4_OUT-1:0][2:0][2:0][RESULT_WIDTH-1:0] kernel5;

  logic signed [CH0_OUT-1:0][RESULT_WIDTH-1:0] bias0;
  logic signed [CH1_OUT-1:0][RESULT_WIDTH-1:0] bias1;
  logic signed [CH2_OUT-1:0][RESULT_WIDTH-1:0] bias2;
  logic signed [CH3_OUT-1:0][RESULT_WIDTH-1:0] bias3;
  logic signed [CH4_OUT-1:0][RESULT_WIDTH-1:0] bias4;
  logic signed [CH5_OUT-1:0][RESULT_WIDTH-1:0] bias5;

  logic signed [CH5_OUT-1:0][RESULT_WIDTH-1:0] final_out;
  logic valid_out;

  logic signed [DATA_WIDTH-1:0] image_mem [0:IMG_WIDTH*IMG_HEIGHT*CH0_IN-1];

  logic signed [DATA_WIDTH-1:0] kernel0_flat [CH0_OUT*CH0_IN*9];
  logic signed [DATA_WIDTH-1:0] kernel1_flat [CH1_OUT*CH0_OUT*9];
  logic signed [DATA_WIDTH-1:0] kernel2_flat [CH2_OUT*CH1_OUT*9];
  logic signed [DATA_WIDTH-1:0] kernel3_flat [CH3_OUT*CH2_OUT*9];
  logic signed [DATA_WIDTH-1:0] kernel4_flat [CH4_OUT*CH3_OUT*9];
  logic signed [DATA_WIDTH-1:0] kernel5_flat [CH5_OUT*CH4_OUT*9];

  logic signed [RESULT_WIDTH-1:0] bias0_flat [CH0_OUT];
  logic signed [RESULT_WIDTH-1:0] bias1_flat [CH1_OUT];
  logic signed [RESULT_WIDTH-1:0] bias2_flat [CH2_OUT];
  logic signed [RESULT_WIDTH-1:0] bias3_flat [CH3_OUT];
  logic signed [RESULT_WIDTH-1:0] bias4_flat [CH4_OUT];
  logic signed [RESULT_WIDTH-1:0] bias5_flat [CH5_OUT];
  int idx, count;

  int of,start_cycle, end_cycle, cycle_count;
  bit started, finished;

  CNNWrapper dut (
    .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .image_in(image_in),
    .kernel0(kernel0), .bias0(bias0),
    .kernel1(kernel1), .bias1(bias1),
    .kernel2(kernel2), .bias2(bias2),
    .kernel3(kernel3), .bias3(bias3),
    .kernel4(kernel4), .bias4(bias4),
    .kernel5(kernel5), .bias5(bias5),
    .final_out(final_out), .valid_out(valid_out)
  );
initial count = 0;
  always #5 clk = ~clk;

  initial begin
  $display("Loading image... ");
  of = $fopen("hardware_out.txt", "w");
    rst_n = 0;
    valid_in = 0;
    #20 rst_n = 1;
    

    $readmemh("mem_files/image_640x640_rgb.mem", image_mem);
    $readmemh("mem_files/kernel0_hex.mem", kernel0_flat);
    $readmemh("mem_files/kernel1_hex.mem", kernel1_flat);
    $readmemh("mem_files/kernel2_hex.mem", kernel2_flat);
    $readmemh("mem_files/kernel3_hex.mem", kernel3_flat);
    $readmemh("mem_files/kernel4_hex.mem", kernel4_flat);
    $readmemh("mem_files/kernel5_test_hex.mem", kernel5_flat);
    $readmemh("mem_files/bias0_hex.mem", bias0_flat);
    $readmemh("mem_files/bias1_hex.mem", bias1_flat);
    $readmemh("mem_files/bias2_hex.mem", bias2_flat);
    $readmemh("mem_files/bias3_hex.mem", bias3_flat);
    $readmemh("mem_files/bias4_hex.mem", bias4_flat);
    $readmemh("mem_files/bias5_hex.mem", bias5_flat);



    idx = 0;
    for (int o = 0; o < CH0_OUT; o++) begin
      bias0[o] = $signed(bias0_flat[o]);
      for (int c = 0; c < CH0_IN; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel0[o][c][i][j] = $signed(kernel0_flat[idx++]);
    end

    idx = 0;
    for (int o = 0; o < CH1_OUT; o++) begin
      bias1[o] = $signed(bias1_flat[o]);
      for (int c = 0; c < CH0_OUT; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel1[o][c][i][j] = $signed(kernel1_flat[idx++]);
    end

    idx = 0;
    for (int o = 0; o < CH2_OUT; o++) begin
      bias2[o] = $signed(bias2_flat[o]);
      for (int c = 0; c < CH1_OUT; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel2[o][c][i][j] = $signed(kernel2_flat[idx++]);
    end

    idx = 0;
    for (int o = 0; o < CH3_OUT; o++) begin
      bias3[o] = $signed(bias3_flat[o]);
      for (int c = 0; c < CH2_OUT; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel3[o][c][i][j] = $signed(kernel3_flat[idx++]);
    end

    idx = 0;
    for (int o = 0; o < CH4_OUT; o++) begin
      bias4[o] = $signed(bias4_flat[o]);
      for (int c = 0; c < CH3_OUT; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel4[o][c][i][j] = $signed(kernel4_flat[idx++]);
    end

    idx = 0;
    for (int o = 0; o < CH5_OUT; o++) begin
      bias5[o] = bias5_flat[o];
      for (int c = 0; c < CH4_OUT; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            kernel5[o][c][i][j] = $signed(kernel5_flat[idx++]);
    end


    for (int y = 0; y < IMG_HEIGHT; y++) begin
      for (int x = 0; x < IMG_WIDTH; x++) begin
        for (int c = 0; c < CH0_IN; c++) begin
          image_in[c] = image_mem[(y * IMG_WIDTH + x) * CH0_IN + c];
        end
        valid_in = 1;
        @(posedge clk);
        if (!started) begin
          started = 1;
          start_cycle = cycle_count;
        end
      end
    end
    valid_in = 0;



  end

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 1;
      if(valid_out) begin
        $fwrite(of, "OUT: %0d %0d", final_out[0], final_out[1]);
        if(!finished) begin
          end_cycle = cycle_count;
            finished = 1;
              $display("Hardware run completed!");
              $display("Start cycle: %0d", start_cycle);
              $display("End cycle: %0d", end_cycle);
              $display("Latency: %0d cycles", end_cycle - start_cycle);
              $display("Latency: %.2f ms", (end_cycle - start_cycle) * CLK_PERIOD_NS * 1e-6); 
          end
        end
      if (finished && cycle_count - end_cycle > 100) begin
        $fclose(of);
          $finish;
      end
    end
  initial begin
    wait (valid_out === 1'b1);
      @(posedge clk);
    $display("Final output: %0d %0d", final_out[0], final_out[1]);
      $finish;
  end


endmodule

