module window_3x3_stream #(
  parameter DATA_WIDTH = 8,
  parameter IMG_WIDTH  = 5,
  parameter IMG_HEIGHT = 5
)(
  input  logic clk,
  input  logic rst_n,
  input  logic valid_in,
  input  logic [DATA_WIDTH-1:0] pixel_in,
  output logic valid_out,
  output logic [DATA_WIDTH-1:0] window [0:2][0:2]
);

  logic [DATA_WIDTH-1:0] linebuf1 [0:IMG_WIDTH-1];
  logic [DATA_WIDTH-1:0] linebuf2 [0:IMG_WIDTH-1];

  logic [DATA_WIDTH-1:0] shift_reg0 [0:2];
  logic [DATA_WIDTH-1:0] shift_reg1 [0:2];
  logic [DATA_WIDTH-1:0] shift_reg2 [0:2];
  
  logic [$clog2(IMG_WIDTH*IMG_HEIGHT):0] pixel_count;

  integer col_counter,row_counter;

  always_ff @(posedge clk or negedge rst_n) begin
  int i;
    if (!rst_n) begin
      col_counter <= 0;
      row_counter <= 0;
      valid_out   <= 0;
      pixel_count <= 0;
    end else if (valid_in) begin
      for (i = 2; i > 0; i = i - 1) begin
        shift_reg0[i] <= shift_reg0[i-1];
        shift_reg1[i] <= shift_reg1[i-1];
        shift_reg2[i] <= shift_reg2[i-1];
      end

      shift_reg0[0] <= pixel_in;
      shift_reg1[0] <= linebuf1[col_counter];
      shift_reg2[0] <= linebuf2[col_counter];

      linebuf2[col_counter] <= linebuf1[col_counter];
      linebuf1[col_counter] <= pixel_in;
      pixel_count <= pixel_count + 1;

      if (col_counter == IMG_WIDTH-1) begin
        col_counter <= 0;
        row_counter <= row_counter +1;
      end
      else
        col_counter <= col_counter + 1;

      if (col_counter >= 2 && row_counter >=2 && pixel_count < (IMG_WIDTH * IMG_HEIGHT))
        valid_out <= 1;
      else
        valid_out <= 0;
    end else begin
      valid_out <= 0;
    end
  end

  always_comb begin
  int j;
    for (j = 0; j < 3; j++) begin
      window[0][j] = shift_reg2[j];
      window[1][j] = shift_reg1[j];
      window[2][j] = shift_reg0[j];
    end
  end

endmodule

