% Prolog POS Tagger, written by Torbjorn Lager


outprob(a,det,0.200).
outprob(the,det,0.200).

outprob(can,n,0.001). 
outprob(man,n,0.002). 

outprob(yesterday,adv,0.005). 

outprob(he,pron,0.070).

outprob(held,vt,0.009).
outprob(fought,vt,0.008).
outprob(fought,vi,0.002).

outprob(taken,vpp,0.0005).
outprob(fought,vpp,0.0003).
outprob(held,vpp,0.001).

outprob(who,rel,0.01).

outprob(was,aux,0.01).

transprob(start,det,0.30). 
transprob(start,vt,0.05). transprob(start,vi,0.05). 
transprob(start,n,0.10). transprob(start,pron,0.30). 

transprob(det,det,0.20).
transprob(det,vi,0.01). transprob(det,vt,0.01). 
transprob(det,n,0.75). transprob(det,pron,0.01).

transprob(vt,det,0.35). transprob(vt,adv,0.04).
transprob(vt,vt,0.01). transprob(vt,vi,0.01).
transprob(vt,n,0.24). transprob(vt,pron,0.35). 

transprob(vi,det,0.05). transprob(vi,adv,0.83).
transprob(vi,vt, 0.01). transprob(vi,vi,0.01).
transprob(vi,n,0.05). transprob(vi,pron,0.05).

transprob(n,det,0.01). transprob(n,n,0.16).
transprob(n,vi,0.16). transprob(n,vt,0.16).
transprob(n,adv,0.06). transprob(n,pron,0.01).
transprob(n,vpp,0.06). transprob(n,rel,0.06).

transprob(pron,det,0.02).
transprob(pron,vi,0.30). transprob(pron,vt,0.30). 
transprob(pron,adv,0.04). transprob(pron,n,0.02).
transprob(pron,pron,0.02).

transprob(vpp,adv,1.0).

transprob(adv,vi,1.0).

transprob(rel,aux,1.0).

transprob(aux,vpp,1.0).


mps(Words,Sequence) :-
    sequences(Words,[1-[start]],PSs),
    max_key(PSs,P-Sequence1),
    reverse(Sequence1,[start|Sequence]).

sequences([],PSs,PSs).
sequences([Word|Words],PSs0,PSs) :-
    findall(PS2,
             (outprob(Word,T2,PL),
             findall(P2-[T2,T1|Ts],
                       (member(P1-[T1|Ts],PSs0),
                       transprob(T1,T2,PT),
                       P2 is PL*PT*P1),
                     PSs),
             max_key(PSs,PS2)),
          PSs1),
    write(PSs1),nl,
    sequences(Words,PSs1,PSs).

max_key(L,K) :- max_key(L,0-[],K).
max_key([],K,K).
max_key([K0-S0|T],KMax-SMax,R) :-
	pick_max(K0-S0,KMax-SMax,KNew-SNew),
	max_key(T,KNew-SNew,R).

pick_max(N1-S1, N2-_S2, N1-S1) :- N1 >= N2, !.
pick_max(_N1-_S1, N2-S2, N2-S2).

