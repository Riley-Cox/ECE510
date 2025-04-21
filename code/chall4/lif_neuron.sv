module lif_neuron #(
    parameter int WIDTH = 16,
    parameter int THRESHOLD = 100,
    parameter int LEAK = 1,
    parameter int REFRACTORY = 5
) (
    input  logic clk,
    input  logic rst,
    input  logic [WIDTH-1:0] input_current,
    output logic spike,
    output logic [WIDTH-1:0] debug_potential  // <-- Added for debug
);

    logic [WIDTH-1:0] membrane_potential;
    logic [$clog2(REFRACTORY+1)-1:0] refractory_cnt;


    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            membrane_potential <= 0;
            refractory_cnt <= 0;
            spike <= 0;
        end else begin
            if (refractory_cnt > 0) begin
                refractory_cnt <= refractory_cnt - 1;
                spike <= 0;
            end else begin
                if (membrane_potential + input_current >= THRESHOLD) begin
                    spike <= 1;
                    membrane_potential <= 0;
                    refractory_cnt <= REFRACTORY;
                end else begin
                    membrane_potential <= (membrane_potential + input_current > LEAK) ?
                                           (membrane_potential + input_current
                                           - LEAK) : 0;
                                                               spike <= 0;
                                                                               end
                                                                                           end
                                                                                                   end
                                                                                                     end
                                                                                                       endmodule

