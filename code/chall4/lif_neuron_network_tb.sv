`timescale 1ns / 1ps

module tb_lif_neuron_network;

    parameter WIDTH = 16;
    parameter NUM_INPUTS = 4;
    parameter NUM_HIDDEN = 3;
    parameter NUM_OUTPUTS = 2;
    parameter THRESHOLD = 100;
    parameter LEAK = 1;
    parameter REFRACTORY = 5;

    logic clk = 0;
    logic rst = 0;
    logic [WIDTH-1:0] input_currents[NUM_INPUTS];
    logic spikes[NUM_OUTPUTS];

    always #5 clk = ~clk;

    lif_neuron_network #(
        .WIDTH(WIDTH),
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_HIDDEN(NUM_HIDDEN),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .THRESHOLD(THRESHOLD),
        .LEAK(LEAK),
        .REFRACTORY(REFRACTORY)
    ) dut (
        .clk(clk),
        .rst(rst),
        .input_currents(input_currents),
        .spikes(spikes)
    );

    initial begin
        $dumpfile("lif_neuron_network.vcd");
        $dumpvars(0, tb_lif_neuron_network);

        rst = 1;
        #20;
        rst = 0;

        input_currents[0] = 16'd120;
        input_currents[1] = 16'd130;
        input_currents[2] = 16'd140;
        input_currents[3] = 16'd150;

        for (int t = 0; t < 300; t++) begin
            @(posedge clk);
            $display("Time %0t: Output Spikes = {%b, %b}", $time, spikes[1],
spikes[0]);
        end

        $finish;
    end

endmodule

