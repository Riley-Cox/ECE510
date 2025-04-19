// Q-learning Update Unit - SystemVerilog Implementation with Fixes and Testbench

module q_learning_update #(
    parameter ROWS = 5,
    parameter COLS = 5,
    parameter ACTIONS = 4,
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 16
)(
    input logic clk,
    input logic rst,

    input logic [2:0] row, col, next_row, next_col,
    input logic [1:0] action,
    input logic signed [DATA_WIDTH-1:0] reward,

    output logic done
);

    typedef logic [6:0] addr_t;
    addr_t addr_sa;

    localparam signed [DATA_WIDTH-1:0] ALPHA = 16'd102; // ~0.1 * 1024
    localparam signed [DATA_WIDTH-1:0] GAMMA = 16'd922; // ~0.9 * 1024

    logic signed [DATA_WIDTH-1:0] q_table [0:ROWS*COLS*ACTIONS-1];
    logic signed [DATA_WIDTH-1:0] q_sa, q_next [0:ACTIONS-1];
    logic signed [DATA_WIDTH-1:0] max_q_next, td_error, delta;
    logic [ADDR_WIDTH-1:0] addr_next [0:ACTIONS-1];

    typedef enum logic [1:0] {IDLE, READ, CALC, WRITE} state_t;
    state_t state, next_state;

    integer i;

    always_ff @(posedge clk) begin
        if (rst) state <= IDLE;
        else     state <= next_state;
    end

    always_comb begin
        case (state)
            IDLE:   next_state = READ;
            READ:   next_state = CALC;
            CALC:   next_state = WRITE;
            WRITE:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            q_sa <= 0;
            for (i = 0; i < ACTIONS; i++) begin
                q_next[i] <= 0;
                addr_next[i] <= 0;
            end
            done <= 0;
        end else begin
            case (state)
                READ: begin
			addr_sa <= addr_t'(32'(row) * 32'(COLS) * 32'(ACTIONS) + 32'(col) * 32'(ACTIONS) + 32'(action));

                    for (i = 0; i < ACTIONS; i++)
                        addr_next[i] <= next_row * COLS * ACTIONS + next_col * ACTIONS + i;
                end

                CALC: begin
                    q_sa <= q_table[addr_sa[6:0]];
                    for (i = 0; i < ACTIONS; i++)
                        q_next[i] <= q_table[addr_next[i]];

                    max_q_next <= q_next[0];
                    for (i = 1; i < ACTIONS; i++)
                        if (q_next[i] > max_q_next)
                            max_q_next <= q_next[i];

                    td_error <= reward + ((GAMMA * max_q_next) >>> 10) - q_sa;
                    delta <= (ALPHA * td_error) >>> 10;
                end

                WRITE: begin
                    q_table[addr_sa] <= q_sa + delta;
                    done <= 1;
                end

                IDLE: done <= 0;
            endcase
        end
    end

endmodule

