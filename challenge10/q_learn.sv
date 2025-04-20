// q_learning_update.sv

module q_learning_update #(
    parameter ROWS = 5,
    parameter COLS = 5,
    parameter ACTIONS = 4,
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = 8
)(
    input logic clk,
    input logic rst,
    input logic [2:0] row,
    input logic [2:0] col,
    input logic [1:0] action,
    input logic [2:0] next_row,
    input logic [2:0] next_col,
    input logic [7:0] reward,
    output logic [ADDR_WIDTH-1:0] addr_sa,
    output logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1],
    output logic [DATA_WIDTH-1:0] q_table_out [0:(ROWS*COLS*ACTIONS)-1]  // Added for debug visibility
);

    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [DATA_WIDTH-1:0] data_t;

    data_t q_table [0:(ROWS*COLS*ACTIONS)-1];
    data_t q_sa;
    data_t q_next [0:ACTIONS-1];
    data_t max_q_next;
    data_t delta;

    logic [DATA_WIDTH-1:0] alpha = 8'd50;  // 0.5 in fixed-point
    logic [DATA_WIDTH-1:0] gamma = 8'd90;  // 0.9 in fixed-point

    enum logic [1:0] {IDLE, FETCH, COMPUTE, UPDATE} state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    state <= FETCH;
                end
                FETCH: begin
                    addr_sa <= addr_t'(row * COLS * ACTIONS + col * ACTIONS + action);
                    for (int i = 0; i < ACTIONS; i++) begin
                        addr_next[i] <= addr_t'(next_row * COLS * ACTIONS + next_col * ACTIONS + i);
                    end
                    state <= COMPUTE;
                end
                COMPUTE: begin
                    q_sa <= q_table[addr_sa];
                    for (int i = 0; i < ACTIONS; i++) begin
                        q_next[i] <= q_table[addr_next[i]];
                    end
                    max_q_next = q_next[0];
                    for (int i = 1; i < ACTIONS; i++) begin
                        if (q_next[i] > max_q_next) max_q_next = q_next[i];
                    end
                    delta <= alpha * (reward + gamma * max_q_next - q_sa) / 100;
                    state <= UPDATE;
                end
                UPDATE: begin
                    q_table[addr_sa] <= q_sa + delta;
                    state <= IDLE;
                end
            endcase
        end
    end

    // Expose internal q_table for debugging
    always_comb begin
        for (int i = 0; i < ROWS*COLS*ACTIONS; i++) begin
            q_table_out[i] = q_table[i];
        end
    end

endmodule

