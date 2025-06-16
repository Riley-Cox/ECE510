module maxpool2x2_stream #(
  parameter DATA_WIDTH = 24,
  parameter IMG_WIDTH = 10,
  parameter IMG_HEIGHT = 10
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  logic [DATA_WIDTH-1:0] pixel_in,
  output logic valid_out,
  output logic [DATA_WIDTH-1:0] pixel_out
);

  logic [DATA_WIDTH-1:0] line_buf0 [0:IMG_WIDTH-1];
  logic [DATA_WIDTH-1:0] line_buf1 [0:IMG_WIDTH-1];
  logic [DATA_WIDTH-1:0] win_buf[0:1][0:1];

  logic [DATA_WIDTH-1:0] max0, max1;

  logic [DATA_WIDTH-1:0] pixel_in_d;
  logic [$clog2(IMG_WIDTH):0] col_count;
  logic [$clog2(IMG_HEIGHT):0] row_count;
  logic pool_ready, pool_fired;

 assign max0 = (win_buf[0][0] > win_buf[0][1]) ? win_buf[0][0] : win_buf[0][1];
  assign max1 = (win_buf[1][0] > win_buf[1][1]) ? win_buf[1][0] : win_buf[1][1];
  assign pixel_out = (max0 > max1) ? max0 : max1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      col_count <= 0;
      row_count <= 0;
      pool_ready <= 0;
      pool_fired <= 0;
      valid_out <= 0;

      for (int i = 0; i < IMG_WIDTH; i++) begin
        line_buf0[i] <= 0;
        line_buf1[i] <= 0;
      end
    end else begin
      valid_out <= 0;

      if (valid_in) begin
        line_buf0[col_count] <= line_buf1[col_count];
        line_buf1[col_count] <= pixel_in;

        if (row_count >= 1)
          pool_ready <= 1;

        if (!pool_fired &&
            pool_ready &&
            row_count[0] && col_count[0] &&
            row_count >= 1 &&
            col_count >= 1 ) begin

          win_buf[0][0] <= line_buf0[col_count - 1];
          win_buf[0][1] <= line_buf0[col_count];
          win_buf[1][0] <= line_buf1[col_count - 1];
          win_buf[1][1] <= line_buf1[col_count];

          valid_out <= 1;
          pool_fired <= 1;
        end

        if (col_count == IMG_WIDTH - 1) begin
          col_count <= 0;
          row_count <= row_count + 1;
          pool_fired <= 0;
        end else begin
          col_count <= col_count + 1;

          if (col_count[0] == 0) // reset fired every 2 cols
            pool_fired <= 0;
        end
      end
    end
  end
endmodule

