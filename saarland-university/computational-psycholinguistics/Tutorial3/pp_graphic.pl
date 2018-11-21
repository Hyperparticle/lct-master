%===============================================================================
%   PP_GRAPHIC.PL
%   Tree pretty-printer.
%
% 1. Find the maximum width of each subtree.
% 2. Add vertical bars so that all the terminals are at the same depth.
% 3. Flatten the tree breadth-wise.
% 4. Print the flattened list.
%===============================================================================

pp_graphic(TREE1) :-
   tree_width(TREE1,TREE2,0),
   add_vertical_bars(TREE2,TREE3),
   pp_flatten([TREE3],FLAT_TREE),!,
   pp_tree(FLAT_TREE),!.


        %-----------------------------------------------------------------------
        % Convert each node in the tree to a functor containing the label on the
        % node, its width, and the maximum width of its subtree, including 2
        % blanks between each subtree. Note: maximum width is instantiated only
        % after the entire subtree has been processed.
        % TREE1 = [s,[np,[n,['John']]],[vp,[v,[is]],[ap,[adj,[crazy]]]]]
        %
        %       s                         pp(s,1,15)
        %    ___|___                __________|__________
        %   np     vp     ==>  pp(np,2,4)          pp(vp,2,9)
        %    |   ___|___            |             ______|______
        %    n   v    ap       pp(n,1,4)     pp(v,1,2)   pp(ap,2,5)
        %    |   |     |            |             |           |
        %  John  is   adj     pp(John,4,4)   pp(is,2,2)  pp(adj,3,5)
        %              |                                      |
        %            crazy                              pp(crazy,5,5)
        %
        %-----------------------------------------------------------------------

tree_width([LEAF],[pp(LEAF,WIDTH,MAXWIDTH)],CURRMAX) :-
   length_atomic(LEAF,WIDTH),
   max(WIDTH,CURRMAX,MAXWIDTH).
tree_width([ROOT,TREE],[pp(ROOT,WIDTH,MAXWIDTH),PPTREE],CURRMAX) :-
   length_atomic(ROOT,WIDTH),
   max(WIDTH,CURRMAX,MAX1),
   tree_width(TREE,PPTREE,MAX1),
   PPTREE = [pp(_,_,MAXWIDTH)|_].
tree_width([ROOT|BODY],[pp(ROOT,WIDTH,MAXWIDTH)|PPBODY],_) :-
   length_atomic(ROOT,WIDTH),
   tree_width1(BODY,PPBODY,0,WIDTHS),
   length(BODY,LEN),
   WIDTHS_PLUS_GAPS is WIDTHS + (LEN-1)*2,      % 2 for the gap
   max(WIDTH,WIDTHS_PLUS_GAPS,MAXWIDTH).

tree_width1([],[],_,0).
tree_width1([TREE|TREES],[PPTREE|PPTREES],CURRMAX,TOTAL_WIDTH) :-
   tree_width(TREE,PPTREE,CURRMAX),
   PPTREE = [pp(_,_,WIDTH)|_],
   tree_width1(TREES,PPTREES,CURRMAX,WIDTHS),
   TOTAL_WIDTH is WIDTHS + WIDTH.


        %-----------------------------------------------------------------------
        % add_vertical_bars(TREE1,TREE2): 
        % change TREE1 to TREE2 (depth=3), adding a parentage indicator to
        % relate daughters with their mothers.
        % TREE1 = [a,[b,[c]],[d,[e,[f]],[g]]]
        % TREE2 = [a,[b,[|,[c]]],[d,[e,[f]],[|,[g]]]]
        %
        %           a                               a,0/1
        %        ___|___                 ___________|__________
        %        b     d        ==>      b,0/1/1              d,0/1/2
        %        |  ___|___              |             _______|_______
        %        c  e     g              |,0/1/1/1     e,0/1/2/1     |,0/1/2/2
        %           |                    |             |             |
        %           f                    c,0/1/1/1/1   f,0/1/2/1/1   g,0/1/2/2/1
        %-----------------------------------------------------------------------

add_vertical_bars(TREE1,TREE2) :- 
   max_depth(TREE1,DEPTH), add_bars(TREE1,1,DEPTH,TREE2,0/1).


max_depth(TREE,DEPTH) :- max_depth1(TREE,1,DEPTH).

max_depth1([_],MAX,MAX).
max_depth1([_|BODY],CUR,MAX) :-
   CUR1 is CUR+1,
   max_depth2(BODY,CUR1,MAX).

max_depth2([],MAX,MAX).
max_depth2([H|T],CUR,MAX) :-
   max_depth1(H,CUR,MAX1),
   max_depth2(T,CUR,MAX2),
   max(MAX1,MAX2,MAX).


add_bars([NODE],DEPTH,DEPTH,[(NODE,POS)],POS).
add_bars([pp(L,W,M)],CUR,DEPTH,[(pp('|',1,M),POS),TREE],POS) :-
   CUR1 is CUR+1,
   add_bars([pp(L,W,M)],CUR1,DEPTH,TREE,POS/1).
add_bars([H|T1],CUR,DEPTH,[(H,POS)|T2],POS) :-
   CUR1 is CUR+1,
   add_bars1(T1,CUR1,DEPTH,T2,POS/1).

add_bars1([],_,_,[],_).
add_bars1([H1|T1],CUR,DEPTH,[H2|T2],POS/I) :-
   add_bars(H1,CUR,DEPTH,H2,POS/I),
   I1 is I+1,
   add_bars1(T1,CUR,DEPTH,T2,POS/I1).


        %-----------------------------------------------------------------------
        % pp_flatten(TREE,FLAT_TREE) 
        % Flatten a tree breadth-wise. Put each level into a list.
        % TREE1 = [[a,[b,[|,[c]]],[d,[e,[f]],[|,[g]]]]]
        % TREE2 = [[a],[b,d],[|,e,|],[c,f,g]]
        %
        % level 0:      a
        %            ___|___
        % level 1:   b     d
        %            |   __|__   =>  [ [a], [b,d], [|,e,|], [c,f,g] ]
        % level 2:   |   e   |
        %            |   |   |
        % level 3:   c   f   g
        %-----------------------------------------------------------------------

pp_flatten([],[]).
pp_flatten(TREE,FLAT_TREE) :-
   pp_flatten1(TREE,FLAT1),
   pp_flatten2(TREE,TREE1),
   pp_flatten(TREE1,FLAT2),
   append([FLAT1],FLAT2,FLAT_TREE).

pp_flatten1([],[]).
pp_flatten1([[H|_]|REST1], [H|REST2]) :-
   pp_flatten1(REST1,REST2).

pp_flatten2([],[]).
pp_flatten2([[_|TREE1]|REST1], FLAT) :-
   pp_flatten2(REST1,FLAT1),
   append(TREE1,FLAT1,FLAT).


        %-----------------------------------------------------------------------
        % pp_tree(TREE): pretty-print the tree
        % TREE:      [ [a], [b,d], [|,e,|], [c,f,g] ]
        % 1. print the node label(s) in the HEAD of the tree:  [a]
        % 2. print the TAIL of the list, aligned under the HEAD node(s):
        %                  [ [b,d], [|,e,|], [c,f,g] ]
        %-----------------------------------------------------------------------

pp_tree([]) :- nl.
pp_tree([HEAD|TAIL]) :-
   pp_labels(HEAD),
   pp_tree1(TAIL,HEAD).

        %-----------------------------------------------------------------------
        % for each label in the list, calculate the following:
        %                            "label"
        %                           ->|   |<- WIDTH
        %          |<--BEFORE_LABEL-->|   |<--AFTER_LABEL----->|
        %          |<---------------MAXWIDTH--------------->|  |
        %                                                 ->|  |<- gap
        %-----------------------------------------------------------------------

pp_labels([]) :- nl.
pp_labels([(pp(LABEL,WIDTH,MAXWIDTH),_)|REST]) :-
   BEFORE_LABEL is (MAXWIDTH + 1)//2 - (WIDTH - 1)//2 - 1,
   AFTER_LABEL is MAXWIDTH - BEFORE_LABEL - WIDTH + 2,           % 2 for the gap
   tab(BEFORE_LABEL),
   write(LABEL),
   tab(AFTER_LABEL),
   pp_labels(REST).

        %-----------------------------------------------------------------------
        %-----------------------------------------------------------------------
pp_tree1([],_) :- nl.
pp_tree1([ROW|ROWS],MOTHERS) :-
   pp_row(MOTHERS,ROW),           % draw line above row
   pp_tree([ROW|ROWS]).

        %-----------------------------------------------------------------------
        % for each mother node, calculate starting position of line,
        % the vertical bar position, and the end of line position.
        %-----------------------------------------------------------------------

pp_row([],_) :- nl.

        %-----------------------------------------------------------------------
        % vertical bar only, no daughters
        %                            BAR_POS
        %                               |
        %                               |<----AFTER_BAR---->|
        %              |<-----------MAXWIDTH------------>|  |
        %                                              ->|  |<- gap
        %-----------------------------------------------------------------------

pp_row([(pp(_,_,MAXWIDTH),_)|MOTHERS],[(pp(_,_,MAXWIDTH),_)|RESTROW]) :-
   BAR_POS is (MAXWIDTH + 1)//2 - 1,
   AFTER_BAR is MAXWIDTH - (BAR_POS + 1) + 2,
   tab(BAR_POS),
   write('|'),
   tab(AFTER_BAR),
   pp_row(MOTHERS,RESTROW).

        %-----------------------------------------------------------------------
        % tree with minimum 2 daughters:
        %
        %        |<-------BAR_POS------>|
        %           ____________________|______________________ 
        %           |<-----LINE1------->|<-------LINE2------->|
        %        label1                                     labeln
        %      ->|  |<- START_POS
        %        |<-----------------END_POS------------------>|
        %        |<-----------------MAXWIDTH-------------------->|
        %                                                      ->|  |<- gap
        %                                      RIGHT_INDENT ->|     |<-
        %
        %-----------------------------------------------------------------------

pp_row([(pp(_,_,MAXWIDTH),DEPTH)|MOTHERS],ROW) :-
   pp_row1(DEPTH,START_POS,END_POS,ROW,RESTROW),
   BAR_POS is (MAXWIDTH + 1)//2 - 1,
   LINE1 is BAR_POS - START_POS,
   LINE2 is END_POS - (BAR_POS + 1),
   RIGHT_INDENT is MAXWIDTH - END_POS + 2,
   tab(START_POS),
   pp_line(LINE1),
   write('|'),
   pp_line(LINE2),
   tab(RIGHT_INDENT),
   pp_row(MOTHERS,RESTROW).


pp_row1(DEPTH,START_POS,END_POS,[(pp(_,_,WIDTH),DEPTH/_)|NODES],REST) :-
   START_POS is (WIDTH - 1)//2,
   pp_row2(WIDTH,DEPTH,0,END_POS,NODES,REST).

pp_row2(WIDTH1,DEPTH,END1,END_POS,[(pp(_,_,WIDTH),DEPTH/_)|NODES],REST) :-
   END2 is END1 + WIDTH1 + 2,            % 2 for the gap
   pp_row2(WIDTH,DEPTH,END2,END_POS,NODES,REST).
pp_row2(WIDTH,_,END1,END_POS,NODES,NODES) :-
   END_POS is END1 + (WIDTH - 1)//2 + 1.
   

        %-------------------------------------------------------------
        % Print a continous line of I underlines.
        %-------------------------------------------------------------

pp_line(0).
pp_line(I) :- write('_'), J is I-1, pp_line(J).


max(I,J,I) :- I >= J.
max(I,J,J) :- I < J.

length_atomic(Atom,Length) :-
	atom(Atom),
	name(Atom,List),
	length(List,Length).


