module lif_neuron_network #(
    parameter int WIDTH = 16,
    parameter int NUM_INPUTS = 4,
    parameter int NUM_HIDDEN = 3,
    parameter int NUM_OUTPUTS = 2,
    parameter int THRESHOLD = 100,
    parameter int LEAK = 1,
    parameter int REFRACTORY = 5
) (
    input logic clk,
    input logic rst,
    input logic [WIDTH-1:0] input_currents [NUM_INPUTS],
    output logic spikes [NUM_OUTPUTS],
    output logic hidden_spikes_debug [NUM_HIDDEN]  // <-- Debug output
);

    logic hidden_spikes [NUM_HIDDEN];
    logic output_spikes [NUM_OUTPUTS];
    logic [WIDTH-1:0] output_inputs [NUM_OUTPUTS];
    logic [WIDTH-1:0] weights [NUM_HIDDEN][NUM_OUTPUTS];

    initial begin
        for (int i = 0; i < NUM_HIDDEN; i++) begin
            for (int j = 0; j < NUM_OUTPUTS; j++) begin
                weights[i][j] = 16'd20;
            end
        end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_HIDDEN; i++) begin : hidden_neurons
            logic [WIDTH-1:0] debug_pot;
            lif_neuron #(
                .WIDTH(WIDTH),
                .THRESHOLD(THRESHOLD),
                .LEAK(LEAK),
                .REFRACTORY(REFRACTORY)
            ) hidden_neuron (
                .clk(clk),
                .rst(rst),
                .input_current(input_currents[i % NUM_INPUTS]),
                .spike(hidden_spikes[i]),
                .debug_potential(debug_pot)  // <-- Connect debug
            );
        end
    endgenerate

    assign hidden_spikes_debug = hidden_spikes;

    always_comb begin
        for (int i = 0; i < NUM_OUTPUTS; i++) begin
            output_inputs[i] = 0;
            for (int j = 0; j < NUM_HIDDEN; j++) begin
                output_inputs[i] += hidden_spikes[j] ? weights[j][i] : 0;
            end
        end
    end

    generate
        for (i = 0; i < NUM_OUTPUTS; i++) begin : output_neurons
            logic [WIDTH-1:0] debug_pot;
            lif_neuron #(
                .WIDTH(WIDTH),
                .THRESHOLD(THRESHOLD),
                .LEAK(LEAK),
                .REFRACTORY(REFRACTORY)
            ) output_neuron (
                .clk(clk),
                .rst(rst),
                .input_current(output_inputs[i]),
                .spike(output_spikes[i]),
                .debug_potential(debug_pot)  // <-- Connect debug
            );
        end
    endgenerate

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < NUM_OUTPUTS; i++) begin
                spikes[i] <= 0;
            end
        end else begin
            for (int i = 0; i < NUM_OUTPUTS; i++) begin
                spikes[i] <= output_spikes[i];
            end
        end
    end
endmodule

