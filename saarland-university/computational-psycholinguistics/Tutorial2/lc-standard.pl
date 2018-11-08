% Left-corner parser for Prolog, with oracle, arc-standard parsing.
%
% Written by: Matthew W. Crocker
% Date: 28 April 2000

% Operator to be used in phrase structure rules.
:- op(1100, xfx, '--->').

% Main predicate for running the test sentences.
do(S) :-
    test(S,Sentence), write(Sentence),nl,
    parse(Sentence).

%Initialization: 
parse(Sentence) :- parse(s,Sentence,[],[s],[]),!.

% Parse a Phrase from S1 to S0 beginning with the current left-corner.
%parse(+Phrase,+S1,S0,+StIn,StOut)
% Phrase - Phrase that needs verification
% S1 - Input sentence
% S0 - Input sentence after it has been read
% StIn - Stack before anything has happened
% StOut - Stack after we've parsed the current word
parse(Phrase, S1, S0, StIn,StOut) :- 
    connects(S1,Word,S2), % Read the current word 
    (Cat ---> [Word]), % Look up word's category in the lexicon
    link(Cat,Phrase), % If can't be linked to category, backtrack
    write(StIn), write(' '), write(S1), nl, 
    lc(Cat, Phrase, S2, S0, StIn,StOut).


% Compose bottom-up and top-down nodes.
% lc(+Phrase,+Phrase,+S0,-S0,+[_|S],-S)
lc(Phrase, Phrase, S0, S0, [_|S],S).

% General rule: arc standard
% lc(+SubPhrase,+SuperPhrase,+S1,S0,+StIn,StOut)
lc(SubPhrase,SuperPhrase, S1, S0, StIn,StOut) :- 
   (Phrase ---> SubPhrase, Right), 
   link(Phrase,SuperPhrase),
   parse(Right, S1, S2,[Right|StIn],St),
   lc(Phrase, SuperPhrase, S2, S0,St,StOut).

% Match next Word from the difference list.
% connects(+[Word|S],-Word,-S)
connects([Word|S], Word, S).

% Grammar

s ---> np, vp.
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

% Oracle
%link(s,s). link(np,np). link(det,det). link(rc,rc). link(sgap,sgap). link(vp,vp).
link(np,s). link(det,np). link(det,s). link(rpro,rc). link(np,sgap). 
link(np,det). link(vt,vp). link(vs,vp). link(det,sgap). link(X,X).
link(vpart,vp).

% Test sentences

test(easy,[the,man,read,the,book]).
test(centre,[the, man, that, the, dog, that, the, cat, chased, bit, read, the, book]).
test(right,[the, man, believes, the, dog, knows, the, cat, said, the, mouse, read, the, book]).
test(left,[the,man,s,dog,s,bowl,fell,down]).
