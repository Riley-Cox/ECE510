module lif_neuron #(
    parameter THRESHOLD = 8,
    parameter LEAK = 1,
    parameter MAX_POTENTIAL = 15
)(
    input logic clk,
    input logic rst,
    input logic [3:0] input_current,
    output logic spike,
    output logic [3:0] potential
);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            potential <= 0;
            spike <= 0;
        end else begin
            if (potential + input_current >= THRESHOLD) begin
                spike <= 1;
                potential <= 0;
            end else begin
                spike <= 0;
                if (potential + input_current >= LEAK)
                    potential <= potential + input_current - LEAK;
                else
                    potential <= 0;
            end
        end
    end

endmodule

