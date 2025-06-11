module maxpool2x2_stream #(
  parameter CONV_KERNEL = 3,
  parameter DATA_WIDTH = 24,
  parameter IMG_WIDTH  = 20,
  parameter IMG_HEIGHT = 20
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  logic [DATA_WIDTH-1:0] pixel_in,
  output logic valid_out,
  output logic [DATA_WIDTH-1:0] pixel_out
);

  localparam int CONV_WIDTH = IMG_WIDTH - CONV_KERNEL + 1;
  localparam int CONV_HEIGHT = IMG_HEIGHT - CONV_KERNEL + 1;
  localparam int NUM_OUTPUTS = (CONV_WIDTH / 2) * (CONV_HEIGHT / 2);
  localparam int TOTAL_POSITIONS = CONV_WIDTH * CONV_HEIGHT;

  logic [15:0] pos_count;
  

  logic [DATA_WIDTH-1:0] line_buf0 [0:IMG_WIDTH-1];
  logic [DATA_WIDTH-1:0] line_buf1 [0:IMG_WIDTH-1];
  logic [DATA_WIDTH-1:0] pixel_in_d;

  logic [$clog2(IMG_WIDTH):0] col_count;
  logic [$clog2(IMG_HEIGHT):0] row_count;

  logic [DATA_WIDTH-1:0] win[0:1][0:1];
  logic pool_ready, pool_fired;

  logic [$clog2(IMG_WIDTH * IMG_HEIGHT)-1:0] pixel_count;
  logic [$clog2(NUM_OUTPUTS+1)-1:0] pool_count;
  logic [DATA_WIDTH-1:0] max0, max1;
  logic drain;
  assign max0 = (win[0][0] > win[0][1]) ? win[0][0] : win[0][1];
  assign max1 = (win[1][0] > win[1][1]) ? win[1][0] : win[1][1];
  assign pixel_out = (CONV_WIDTH == 1 && CONV_HEIGHT == 1) ? win[0][0] : ((max0 > max1) ? max0 : max1);


 
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      pool_count <= 0;
    else if (valid_out)
      pool_count <= pool_count + 1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      pixel_count <= 0;
    else if (valid_in)
      pixel_count <= pixel_count + 1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) 
      drain <= 0;
    else if (pixel_count == (CONV_HEIGHT * CONV_WIDTH))
      drain <= 1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      pixel_in_d <= 0;
    end
    else if (valid_in) begin
      pixel_in_d <= pixel_in;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      col_count   <= 0;
      row_count   <= 0;
      pool_ready  <= 0;
      pool_fired  <= 0;
      valid_out   <= 0;

      for (int i = 0; i < CONV_WIDTH; i++) begin
        line_buf0[i] <= 0;
        line_buf1[i] <= 0;
      end
    end else begin
      valid_out <= 0;

      if (valid_in || drain || pos_count < TOTAL_POSITIONS) begin
        line_buf0[col_count] <= line_buf1[col_count];
        line_buf1[col_count] <= pixel_in_d;
        pos_count <= pos_count + 1;

        if (row_count >= 1)
          pool_ready <= 1;

        if (col_count == CONV_WIDTH - 1) begin
          col_count <= 0;
          row_count <= row_count + 1;
        end else begin
          col_count <= col_count + 1;
        end
      end
      if ((CONV_WIDTH == 1 && CONV_HEIGHT == 1) &&
          !pool_fired && pool_count == 0 && valid_in) begin
          valid_out <= 1;
          pool_fired <= 1;
          pool_count <= 1;
          win[0][0] <= pixel_in;
        end

      else if ((pool_ready || drain) &&
          pixel_count >= (CONV_WIDTH *2) &&
          row_count >= 1 &&
          col_count >= 1 &&
          col_count[0] && row_count[0] &&  // col and row are odd
          !pool_fired &&
          pool_count < NUM_OUTPUTS) begin

        win[0][0] <= line_buf0[col_count - 1];
        win[0][1] <= line_buf0[col_count];
        win[1][0] <= line_buf1[col_count - 1];
        win[1][1] <= line_buf1[col_count];

        valid_out <= 1;
        pool_fired <= 1;
      end

      if (col_count[0] == 0 && row_count[0])
        pool_fired <= 0;
    end
  end
endmodule

