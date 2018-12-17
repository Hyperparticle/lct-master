% Prolog POS Tagger, written by Torbjorn Lager


outprob(a,det,0.300).
outprob(can,aux,0.010).
outprob(can,v,0.005).
outprob(can,n,0.001). 
outprob(he,pron,0.070).

transprob(start,det,0.30). transprob(start,aux,0.20).
transprob(start,v,0.10). transprob(start,n,0.10).
transprob(start,pron,0.30). transprob(det,det,0.20).
transprob(det,aux,0.01). transprob(det,v,0.01).
transprob(det,n,0.77). transprob(det,pron,0.01).
transprob(aux,det,0.18). transprob(aux,aux,0.10).
transprob(aux,v,0.50). transprob(aux,n,0.01).
transprob(aux,pron,0.21). transprob(v,det,0.36).
transprob(v,aux,0.01). transprob(v,v,0.01).
transprob(v,n,0.26). transprob(v,pron,0.36).
transprob(n,det,0.01). transprob(n,aux,0.25).
transprob(n,v,0.39). transprob(n,n,0.34).
transprob(n,pron,0.01). transprob(pron,det,0.01).
transprob(pron,aux,0.45). transprob(pron,v,0.52).
transprob(pron,n,0.01). transprob(pron,pron,0.01).

most_probable_sequence(Words,Ss) :-
	findall(PS,sequence(Words,1-[start],PS),PSs),
	max_key(PSs,P-Ss1),
	reverse(Ss1,[start|Ss]).

sequence([],PSs,PSs). 
sequence([Word|Words],P1-[S1|Ss],PSs) :-
	outprob(Word,S2,Po),
	transprob(S1,S2,Pt),
	P2 is Po*Pt*P1,
      	write(P2-[S2,S1|Ss]),nl,
	sequence(Words,P2-[S2,S1|Ss],PSs).

max_key(L,K) :- max_key(L,0-[],K).
max_key([],K,K).
max_key([K0-S0|T],KMax-SMax,R) :-
	pick_max(K0-S0,KMax-SMax,KNew-SNew),
	max_key(T,KNew-SNew,R).

pick_max(N1-S1, N2-_S2, N1-S1) :- N1 >= N2, !.
pick_max(_N1-_S1, N2-S2, N2-S2).

