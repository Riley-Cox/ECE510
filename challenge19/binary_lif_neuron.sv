module binary_lif_neuron #(
    parameter int THRESHOLD = 4,
    parameter int LEAK = 1,
    parameter int MAX_MEM = 15
)(
    input  logic clk,
    input  logic rst,
    input  logic in,
    output logic spike_out
);

    logic [$clog2(MAX_MEM+1)-1:0] membrane_potential;
    logic [$clog2(MAX_MEM+1)-1:0] next_potential;
    logic next_spike;

    // Combinational logic: compute next potential and spike
    always_comb begin
        next_potential = membrane_potential;

        // Leak
        if (next_potential >= LEAK)
            next_potential -= LEAK;
        else
            next_potential = 0;

        // Integrate input
        if (in)
            next_potential += 1;

        // Clamp
        if (next_potential > MAX_MEM)
            next_potential = MAX_MEM;

        // Spike check
        next_spike = (next_potential >= THRESHOLD);
    end

    // Sequential logic: register state and output
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            membrane_potential <= 0;
            spike_out <= 0;
        end else begin
            spike_out <= next_spike;
            if (next_spike)
                membrane_potential <= 0;
            else
                membrane_potential <= next_potential;
        end
    end

endmodule

