module spi_slave (
    input  logic clk,           // System clock
    input  logic rst_n,         // Active-low reset
    input  logic sclk,          // SPI clock
    input  logic mosi,          // Master Out Slave In
    output logic miso,          // Master In Slave Out
    input  logic cs_n,          // Active-low chip select
    output logic [7:0] rx_data, // Output received byte
    input  logic [7:0] tx_data, // Input byte to transmit
    output logic done           // Pulse high when byte is received
);

    logic [7:0] shift_reg_rx;
    logic [7:0] shift_reg_tx;
    logic [2:0] bit_cnt;

    // Drive MISO with MSB of transmit shift register
    assign miso = (!cs_n) ? shift_reg_tx[7] : 1'b0;

    // Load tx_data at start of new byte (bit_cnt == 0)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_tx <= 8'd0;
        end else if (!cs_n && bit_cnt == 0) begin
            shift_reg_tx <= tx_data;
        end
    end

    // Receive logic: sample MOSI on rising edge of SCLK
    always_ff @(posedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_rx <= 8'd0;
            bit_cnt <= 3'd0;
            done <= 1'b0;
            rx_data <= 8'd0;
        end else if (!cs_n) begin
            shift_reg_rx <= {shift_reg_rx[6:0], mosi};
            bit_cnt <= bit_cnt + 1;

            if (bit_cnt == 3'd7) begin
                rx_data <= {shift_reg_rx[6:0], mosi};
                done <= 1'b1;
                bit_cnt <= 3'd0;
            end else begin
                done <= 1'b0;
            end
        end else begin
            bit_cnt <= 3'd0;
            done <= 1'b0;
        end
    end

    // Transmit logic: shift MISO on falling edge of SCLK
    always_ff @(negedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_tx <= 8'd0;
        end else if (!cs_n) begin
            shift_reg_tx <= {shift_reg_tx[6:0], 1'b0};
        end
    end

    // VCD dump for waveform viewing
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, spi_slave);
    end

endmodule

