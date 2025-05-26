module binary_lif_neuron #(
    parameter integer THRESHOLD = 5,
    parameter integer LEAK = 1,
    parameter integer MAX_MEM = 15  // Optional saturation
)(
    input logic clk,
    input logic rst,
    input logic in,               // Binary spike input (0 or 1)
    output logic spike_out        // Binary output spike
);

    logic [$clog2(MAX_MEM+1)-1:0] membrane_potential;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            membrane_potential <= 0;
            spike_out <= 0;
        end else begin
            // Leak first
            if (membrane_potential >= LEAK)
                membrane_potential <= membrane_potential - LEAK;
            else
                membrane_potential <= 0;

            // Integrate input
            if (in)
                membrane_potential <= membrane_potential + 1;

            // Cap membrane potential to avoid overflow
            if (membrane_potential > MAX_MEM)
                membrane_potential <= MAX_MEM;

            // Check for spike
            if (membrane_potential >= THRESHOLD) begin
                spike_out <= 1;
                membrane_potential <= 0;  // Reset after firing
            end else begin
                spike_out <= 0;
            end
        end
    end

endmodule

