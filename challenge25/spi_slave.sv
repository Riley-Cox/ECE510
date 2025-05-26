module spi_slave (
    input  logic clk,
    input  logic rst_n,
    input  logic sclk,
    input  logic mosi,
    output logic miso,
    input  logic cs_n,
    output logic [7:0] rx_data,
    input  logic [7:0] tx_data
);

    logic [7:0] shift_reg_tx,shift_reg_rx;
    logic [2:0] bit_cnt;
    logic done;
    logic sclk_prev;
    assign miso = (!cs_n) ? shift_reg_tx[7] : 1'b0;  // Always drive something

    always_ff @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		sclk_prev = 0;
	end
	else begin 
		sclk_prev <= sclk;
	end
end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_rx <= 8'd0;
	    bit_cnt <= 3'd0;
	    done <= 1'b0;
        end
	else begin
	done <= 0;
	if (!cs_n && sclk_prev && !sclk) begin
	    shift_reg_rx <= {shift_reg_rx[6:0], 1'b0};
	    bit_cnt <= bit_cnt + 1;
		if (bit_cnt == 3'd7) begin
		rx_data <= {shift_reg_rx[6:0], mosi};
		done <= 1;
		bit_cnt <= 0;
		end
	end
	end	
    end
endmodule

