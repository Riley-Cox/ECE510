module lif_neuron_fixedpoint #(
    parameter THRESHOLD = 8'd64,     // threshold in Q4.4 format (e.g., 4.0 = 64)
    parameter LEAK_FACTOR = 8'd12    // lambda in Q4.4 (e.g., 0.75 = 12)
)(
    input  logic clk,
    input  logic rst,
    input  logic input_spike,          // I(t): binary input
    output logic spike,                // S(t): spike output
    output logic [7:0] potential       // P(t): Q4.4 fixed-point
);

    logic [15:0] scaled_potential;
    logic [7:0] new_potential;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            potential <= 0;
            spike <= 0;
        end else begin
            // 1. Apply leak: P(t) = lambda * P(t-1)
            scaled_potential = (potential * LEAK_FACTOR) >> 4;

            // 2. Add input (0 or 1) in Q4.4 format → I(t) = 16
            new_potential = scaled_potential[7:0] + (input_spike ? 8'd16 : 8'd0);

            // 3. Threshold & spike
            if (new_potential >= THRESHOLD) begin
                spike <= 1;
                potential <= 0;  // reset
            end else begin
                spike <= 0;
                potential <= new_potential;
            end
        end
    end

endmodule

