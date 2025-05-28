

module maxpool2x2_stream #(
  parameter DATA_WIDTH = 24,
  parameter IMG_WIDTH = 4,
  parameter IMG_HEIGHT = 4
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  logic signed [DATA_WIDTH-1:0] pixel_in,
  output logic valid_out,
  output logic signed [DATA_WIDTH-1:0] pixel_out
);

  logic signed [DATA_WIDTH-1:0] line_buf [0:1][0:IMG_WIDTH-1];
  logic signed [DATA_WIDTH-1:0] win_buf [0:1][0:1];




  logic [15:0] row_count;
  logic [15:0] col_count,col_count_d;
  logic toggle;
  logic toggle_d;
  logic next_valid, pool_ready;

  logic load_window;
  int i;

  logic signed [DATA_WIDTH-1:0] max0, max1;
  assign max0 = (win_buf[0][0] > win_buf[0][1]) ? win_buf[0][0] : win_buf[0][1];
  assign max1 = (win_buf[1][0] > win_buf[1][1]) ? win_buf[1][0] : win_buf[1][1];
  assign pixel_out = (max0 > max1) ? max0 : max1;


  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      row_count <= 0;
      col_count <= 0;
      toggle <= 0;
      valid_out <= 0;
      pool_ready <= 0;
      for (i = 0; i < IMG_WIDTH; i++) begin
        line_buf[0][i] <= 0;
        line_buf[1][i] <= 0;
      end
      win_buf[0][0] <= 0;
      win_buf[0][1] <= 0;
      win_buf[1][0] <= 0;
      win_buf[1][1] <= 0;
    
    end else begin
      valid_out <= 0;
      load_window <= 0;
      if (valid_in) begin
        line_buf[toggle][col_count] <= pixel_in;
        toggle_d <= toggle;
        col_count_d <= col_count;
        if (row_count >= 2)
          pool_ready <= 1;
      if (toggle_d && col_count_d >= 1 && valid_in && (col_count_d % 2 == 1) && pool_ready) begin
        win_buf[0][0] <= line_buf[~toggle_d][col_count_d - 1];
        win_buf[0][1] <= line_buf[~toggle_d][col_count_d];
        win_buf[1][0] <= line_buf[toggle_d][col_count_d - 1];
        win_buf[1][1] <= line_buf[toggle_d][col_count_d];
        valid_out <= 1;
      end
      if (col_count == IMG_WIDTH - 1) begin
        col_count <= 0;
        row_count <= row_count + 1;
        toggle <= ~toggle;
      end else begin 
        col_count <= col_count + 1;
      end
    end
  end
  if(valid_out) begin
    $display("POOLING WINDOW: [%0d %0d] [%0d %0d] => %0d",win_buf[0][0], win_buf[0][1], win_buf[1][0], win_buf[1][1], pixel_out); 
  end
end

endmodule
      

