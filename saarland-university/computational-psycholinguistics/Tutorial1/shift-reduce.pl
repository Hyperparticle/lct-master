% Left-corner parser for Prolog, with oracle, including predicate for
% arc-eager as well as arc-standard parsing.
%
% Written by: Matthew W. Crocker
% Date: 28 April 2000

% Operator to be used in phrase structure rules.

:- op(1100, xfx, '--->').

% Main predicate for running the test sentences.
do(S) :-
    test(S,Sentence),
    write(Sentence),nl,
    parse([],Sentence).

% Done
parse([s],[]) :- write('[] []'),nl,!.

% Reduce
parse([Y,X|Rest],String) :-
    (LHS ---> X,Y),
				write([LHS|Rest]),write(String),nl,
    parse([LHS|Rest],String).

% Shift
parse(Stack,[Word|Rest]) :-
    (Cat ---> [Word]),
    write([Cat|Stack]),write(Rest),nl,
    parse([Cat|Stack],Rest).

% Grammar

s ---> np, vp.
np ---> possnp, np.
np ---> det, n.
np ---> np, rc.
det ---> np, poss.
rc ---> rpro, sgap.
sgap ---> np, vt.
vp ---> vt, np.
vp ---> vs, s.
vp ---> vpart, part.

% Lexicon

det ---> [the].
rpro ---> [that].
poss ---> [s].
part ---> [down].

n ---> [man].
n ---> [dog].
n ---> [cat].
n ---> [mouse].
n ---> [book].
n ---> [bowl].

vt ---> [read].
vt ---> [chased].
vt ---> [bit].

vs ---> [knows].
vs ---> [believes].
vs ---> [said].

vpart ---> [fell].

% Test sentences

test(easy,[the,man,read,the,book]).
test(centre,[the, man, that, the, dog, that, the, cat, chased, bit, read, the, book]).
test(right,[the, man, believes, the, dog, knows, the, cat, said, the, mouse, read, the, book]).
test(left,[the,man,s,dog,s,bowl,fell,down]).







