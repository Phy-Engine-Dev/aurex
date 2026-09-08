module snake6x6 (
    input  clk,
    input  rst_n,
    input  left,
    input  right,
    input  rotate,

    output reg game_over,
    output [5:0] row0,
    output [5:0] row1,
    output [5:0] row2,
    output [5:0] row3,
    output [5:0] row4,
    output [5:0] row5
);

    // ============================
    // 6x6 locked board
    // ============================
    reg [5:0] b0, b1, b2, b3, b4, b5;

    // ============================
    // active piece meta
    //  ptype: 0=I 1=O 2=T 3=L
    //  prot : 0..3 (rotation)
    //  px/py: top-left origin for the piece's local bounding box
    // ============================
    reg [1:0] ptype;
    reg [1:0] prot;
    reg [2:0] px;
    reg [2:0] py;
    reg [1:0] next_type;

    // FSM: 0=PLAY, 1=CLEAR, 2=SPAWN
    reg [1:0] state;
    reg [2:0] scan;

    wire [5:0] FULL;
    assign FULL = 6'b111111;

    // ============================
    // current piece shape (4 local rows, LSB is left)
    // ============================
    reg [3:0] sh0, sh1, sh2, sh3;
    reg [2:0] sh_w, sh_h;

    always @(*) begin
        sh0 = 4'b0000; sh1 = 4'b0000; sh2 = 4'b0000; sh3 = 4'b0000;
        sh_w = 3'd0; sh_h = 3'd0;
        if (ptype == 2'd0) begin
            // I
            if (prot == 2'd0 || prot == 2'd2) begin
                sh0 = 4'b1111; sh_w = 3'd4; sh_h = 3'd1;
            end else begin
                sh0 = 4'b0001; sh1 = 4'b0001; sh2 = 4'b0001; sh3 = 4'b0001;
                sh_w = 3'd1; sh_h = 3'd4;
            end
        end else if (ptype == 2'd1) begin
            // O
            sh0 = 4'b0011; sh1 = 4'b0011;
            sh_w = 3'd2; sh_h = 3'd2;
        end else if (ptype == 2'd2) begin
            // T
            if (prot == 2'd0) begin
                sh0 = 4'b0111; sh1 = 4'b0010; sh_w = 3'd3; sh_h = 3'd2;
            end else if (prot == 2'd1) begin
                sh0 = 4'b0010; sh1 = 4'b0011; sh2 = 4'b0010; sh_w = 3'd2; sh_h = 3'd3;
            end else if (prot == 2'd2) begin
                sh0 = 4'b0010; sh1 = 4'b0111; sh_w = 3'd3; sh_h = 3'd2;
            end else begin
                sh0 = 4'b0001; sh1 = 4'b0011; sh2 = 4'b0001; sh_w = 3'd2; sh_h = 3'd3;
            end
        end else begin
            // L
            if (prot == 2'd0) begin
                sh0 = 4'b0111; sh1 = 4'b0100; sh_w = 3'd3; sh_h = 3'd2;
            end else if (prot == 2'd1) begin
                sh0 = 4'b0001; sh1 = 4'b0001; sh2 = 4'b0011; sh_w = 3'd2; sh_h = 3'd3;
            end else if (prot == 2'd2) begin
                sh0 = 4'b0001; sh1 = 4'b0111; sh_w = 3'd3; sh_h = 3'd2;
            end else begin
                sh0 = 4'b0011; sh1 = 4'b0001; sh2 = 4'b0001; sh_w = 3'd2; sh_h = 3'd3;
            end
        end
    end

    // ============================
    // active piece masks in board coordinates
    // ============================
    wire active_en;
    assign active_en = (state == 2'd0);

    wire [5:0] s0, s1, s2, s3;
    assign s0 = ({2'b00, sh0} << px);
    assign s1 = ({2'b00, sh1} << px);
    assign s2 = ({2'b00, sh2} << px);
    assign s3 = ({2'b00, sh3} << px);

    reg [5:0] ar0, ar1, ar2, ar3, ar4, ar5;
    always @(*) begin
        ar0 = 6'd0; ar1 = 6'd0; ar2 = 6'd0; ar3 = 6'd0; ar4 = 6'd0; ar5 = 6'd0;
        if (active_en) begin
            if (py == 3'd0) begin
                ar0 = s0; ar1 = s1; ar2 = s2; ar3 = s3;
            end else if (py == 3'd1) begin
                ar1 = s0; ar2 = s1; ar3 = s2; ar4 = s3;
            end else if (py == 3'd2) begin
                ar2 = s0; ar3 = s1; ar4 = s2; ar5 = s3;
            end else if (py == 3'd3) begin
                ar3 = s0; ar4 = s1; ar5 = s2;
            end else if (py == 3'd4) begin
                ar4 = s0; ar5 = s1;
            end else begin
                ar5 = s0;
            end
        end
    end

    // ============================
    // collision helpers
    // ============================
    wire [5:0] ar_or;
    assign ar_or = ar0 | ar1 | ar2 | ar3 | ar4 | ar5;

    wire left_edge, right_edge;
    assign left_edge  = ((ar_or & 6'b000001) != 0);
    assign right_edge = ((ar_or & 6'b100000) != 0);

    wire collide_left, collide_right;
    assign collide_left =
        ((((ar0>>1)&b0)!=0) | (((ar1>>1)&b1)!=0) | (((ar2>>1)&b2)!=0) |
         (((ar3>>1)&b3)!=0) | (((ar4>>1)&b4)!=0) | (((ar5>>1)&b5)!=0));

    assign collide_right =
        ((((ar0<<1)&b0)!=0) | (((ar1<<1)&b1)!=0) | (((ar2<<1)&b2)!=0) |
         (((ar3<<1)&b3)!=0) | (((ar4<<1)&b4)!=0) | (((ar5<<1)&b5)!=0));

    wire can_left, can_right;
    assign can_left  = (~left_edge)  & (~collide_left);
    assign can_right = (~right_edge) & (~collide_right);

    wire hit_bottom;
    assign hit_bottom = (ar5 != 0);

    wire collide_down;
    assign collide_down =
        ((ar4 & b5)!=0) | ((ar3 & b4)!=0) | ((ar2 & b3)!=0) |
        ((ar1 & b2)!=0) | ((ar0 & b1)!=0);

    wire can_down;
    assign can_down = (~hit_bottom) & (~collide_down);

    // ============================
    // rotate candidate
    // ============================
    wire [1:0] prot_rot;
    assign prot_rot = prot + 2'd1;

    reg [3:0] rsh0, rsh1, rsh2, rsh3;
    reg [2:0] rsh_w, rsh_h;
    always @(*) begin
        rsh0 = 4'b0000; rsh1 = 4'b0000; rsh2 = 4'b0000; rsh3 = 4'b0000;
        rsh_w = 3'd0; rsh_h = 3'd0;
        if (ptype == 2'd0) begin
            if (prot_rot == 2'd0 || prot_rot == 2'd2) begin
                rsh0 = 4'b1111; rsh_w = 3'd4; rsh_h = 3'd1;
            end else begin
                rsh0 = 4'b0001; rsh1 = 4'b0001; rsh2 = 4'b0001; rsh3 = 4'b0001;
                rsh_w = 3'd1; rsh_h = 3'd4;
            end
        end else if (ptype == 2'd1) begin
            rsh0 = 4'b0011; rsh1 = 4'b0011;
            rsh_w = 3'd2; rsh_h = 3'd2;
        end else if (ptype == 2'd2) begin
            if (prot_rot == 2'd0) begin
                rsh0 = 4'b0111; rsh1 = 4'b0010; rsh_w = 3'd3; rsh_h = 3'd2;
            end else if (prot_rot == 2'd1) begin
                rsh0 = 4'b0010; rsh1 = 4'b0011; rsh2 = 4'b0010; rsh_w = 3'd2; rsh_h = 3'd3;
            end else if (prot_rot == 2'd2) begin
                rsh0 = 4'b0010; rsh1 = 4'b0111; rsh_w = 3'd3; rsh_h = 3'd2;
            end else begin
                rsh0 = 4'b0001; rsh1 = 4'b0011; rsh2 = 4'b0001; rsh_w = 3'd2; rsh_h = 3'd3;
            end
        end else begin
            if (prot_rot == 2'd0) begin
                rsh0 = 4'b0111; rsh1 = 4'b0100; rsh_w = 3'd3; rsh_h = 3'd2;
            end else if (prot_rot == 2'd1) begin
                rsh0 = 4'b0001; rsh1 = 4'b0001; rsh2 = 4'b0011; rsh_w = 3'd2; rsh_h = 3'd3;
            end else if (prot_rot == 2'd2) begin
                rsh0 = 4'b0001; rsh1 = 4'b0111; rsh_w = 3'd3; rsh_h = 3'd2;
            end else begin
                rsh0 = 4'b0011; rsh1 = 4'b0001; rsh2 = 4'b0001; rsh_w = 3'd2; rsh_h = 3'd3;
            end
        end
    end

    wire [2:0] px_rot;
    assign px_rot = (px > (3'd6 - rsh_w)) ? (3'd6 - rsh_w) : px;

    wire [5:0] rs0, rs1, rs2, rs3;
    assign rs0 = ({2'b00, rsh0} << px_rot);
    assign rs1 = ({2'b00, rsh1} << px_rot);
    assign rs2 = ({2'b00, rsh2} << px_rot);
    assign rs3 = ({2'b00, rsh3} << px_rot);

    reg [5:0] rr0, rr1, rr2, rr3, rr4, rr5;
    always @(*) begin
        rr0 = 6'd0; rr1 = 6'd0; rr2 = 6'd0; rr3 = 6'd0; rr4 = 6'd0; rr5 = 6'd0;
        if (py == 3'd0) begin
            rr0 = rs0; rr1 = rs1; rr2 = rs2; rr3 = rs3;
        end else if (py == 3'd1) begin
            rr1 = rs0; rr2 = rs1; rr3 = rs2; rr4 = rs3;
        end else if (py == 3'd2) begin
            rr2 = rs0; rr3 = rs1; rr4 = rs2; rr5 = rs3;
        end else if (py == 3'd3) begin
            rr3 = rs0; rr4 = rs1; rr5 = rs2;
        end else if (py == 3'd4) begin
            rr4 = rs0; rr5 = rs1;
        end else begin
            rr5 = rs0;
        end
    end

    wire rotate_in_bounds;
    assign rotate_in_bounds = (py + rsh_h) <= 3'd6;

    wire rotate_collide;
    assign rotate_collide =
        ((rr0 & b0)!=0) | ((rr1 & b1)!=0) | ((rr2 & b2)!=0) | ((rr3 & b3)!=0) | ((rr4 & b4)!=0) | ((rr5 & b5)!=0);

    wire can_rotate;
    assign can_rotate = active_en & rotate_in_bounds & (~rotate_collide);

    // ============================
    // spawn (deterministic first piece, then $random)
    // ============================
    reg [2:0] spx;
    always @(*) begin
        if (next_type == 2'd0) spx = 3'd1;       // I (width 4)
        else if (next_type == 2'd1) spx = 3'd2;  // O (width 2)
        else spx = 3'd1;                         // T/L (width 3)
    end

    wire [5:0] sp0, sp1;
    assign sp0 = (next_type == 2'd0) ? (6'b001111 << spx) :
                 (next_type == 2'd1) ? (6'b000011 << spx) :
                 (next_type == 2'd2) ? (6'b000111 << spx) : (6'b000111 << spx);
    assign sp1 = (next_type == 2'd1) ? (6'b000011 << spx) :
                 (next_type == 2'd2) ? (6'b000010 << spx) :
                 (next_type == 2'd3) ? (6'b000100 << spx) : 6'd0;

    wire spawn_collide;
    assign spawn_collide = ((sp0 & b0)!=0) | ((sp1 & b1)!=0);

    // ============================
    // main sequential
    // ============================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b0 <= 0; b1 <= 0; b2 <= 0; b3 <= 0; b4 <= 0; b5 <= 0;
            ptype <= 2'd0;
            prot <= 2'd0;
            px <= 3'd1;
            py <= 3'd0;
            next_type <= 2'd0;  // deterministic first spawn
            game_over <= 1'b0;
            state <= 2'd2;  // SPAWN
            scan <= 3'd5;
        end else if (!game_over) begin
            // CLEAR
            if (state == 2'd1) begin
                if (scan == 3'd5) begin
                    if (b5 == FULL) begin b5 <= b4; b4 <= b3; b3 <= b2; b2 <= b1; b1 <= b0; b0 <= 0; end
                    else scan <= 3'd4;
                end else if (scan == 3'd4) begin
                    if (b4 == FULL) begin b4 <= b3; b3 <= b2; b2 <= b1; b1 <= b0; b0 <= 0; end
                    else scan <= 3'd3;
                end else if (scan == 3'd3) begin
                    if (b3 == FULL) begin b3 <= b2; b2 <= b1; b1 <= b0; b0 <= 0; end
                    else scan <= 3'd2;
                end else if (scan == 3'd2) begin
                    if (b2 == FULL) begin b2 <= b1; b1 <= b0; b0 <= 0; end
                    else scan <= 3'd1;
                end else if (scan == 3'd1) begin
                    if (b1 == FULL) begin b1 <= b0; b0 <= 0; end
                    else scan <= 3'd0;
                end else begin
                    if (b0 == FULL) b0 <= 0;
                    state <= 2'd2;  // SPAWN
                end
            end

            // SPAWN
            else if (state == 2'd2) begin
                if (spawn_collide) begin
                    game_over <= 1'b1;
                end else begin
                    ptype <= next_type;
                    prot <= 2'd0;
                    px <= spx;
                    py <= 3'd0;
                    next_type <= $random;
                    state <= 2'd0;  // PLAY
                end
            end

            // PLAY
            else begin
                if (rotate && can_rotate) begin
                    prot <= prot_rot;
                    px <= px_rot;
                end else if (left && can_left) begin
                    px <= px - 3'd1;
                end else if (right && can_right) begin
                    px <= px + 3'd1;
                end else if (can_down) begin
                    py <= py + 3'd1;
                end else begin
                    // lock
                    b0 <= b0 | ar0;
                    b1 <= b1 | ar1;
                    b2 <= b2 | ar2;
                    b3 <= b3 | ar3;
                    b4 <= b4 | ar4;
                    b5 <= b5 | ar5;
                    state <= 2'd1;
                    scan <= 3'd5;
                end
            end
        end
    end

    // ============================
    // output
    // ============================
    assign row0 = game_over ? 6'b100001 : (b0 | ar0);
    assign row1 = game_over ? 6'b010010 : (b1 | ar1);
    assign row2 = game_over ? 6'b001100 : (b2 | ar2);
    assign row3 = game_over ? 6'b001100 : (b3 | ar3);
    assign row4 = game_over ? 6'b010010 : (b4 | ar4);
    assign row5 = game_over ? 6'b100001 : (b5 | ar5);

endmodule
