%%
% A simple implementation of the Competition-Integration model (CIM). See:
%
% McRae, K., Spivey-Knowlton, M., and Tanenhaus, M. (1998). Modeling the
%   influence of thematic fit (and other constraints) in on-line sentence
%   comprehension. Journal of Memory and Language, 38(3):283–312.
%
% Green, M. and Mitchell, D. (2006). Absence of real evidence against
%   competition during syntactic ambiguity resolution. Journal of Memory and
%   Language, 55(1):1–17.
%
% (c) 2015 Harm Brouwer <me@hbrouwer.eu>
%%

:- module(cim,[
        run_model/0
]).

% Flags whether model state summaries are given after each cycle.

% Note: this is now specified in model files that import this module.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                M O D E L   S P E C I F I C A T I O N              %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% delta(-Delta)
%
%  The delta parameter, used to control the activation threshold.

% Note: this is now specified in model files that import this module.

%% input_node(-Constraint,-Alternative,-Weight,-Activation).
%
%  This predicate is used to specify the model layout. Each line specifies
%  a Constraint for an Alternative, the weight for this Constraint, and its
%  initial Activation.
%
%  Each constraint should be fully specified for each alternative, and
%  weights should ideally sum to 1.0. Note: this will not be checked,
%  leaving you free to play around with different configurations.

% Note: this is now specified in model files that import this module.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                       M O D E L   C O D E                         %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run_model :-
        setof(Al,C^W^Ac^input_node(C,Al,W,Ac),Als),    % alternatives
        setof(C,Al^W^Ac^input_node(C,Al,W,Ac),Cs),     % constraints
        findall((C,Al,W,Ac),input_node(C,Al,W,Ac),Is), % input nodes
        process_cycles(Als,Cs,Is,1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Processing cycles %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The main processing loop ...
process_cycles(Als,Cs,Is,Cyc) :-
        process_cycle(Als,Cs,Is,IntAls,UpdIs),       % process current cycle
        (  threshold_reached(IntAls,Cyc)             % Q: is competition over?
        -> format_cycle_summary(IntAls,UpdIs,Cyc),   % yes: stop processing
           format_processing_summary(IntAls,Cyc)
        ;  (  flag_verbose
           -> format_cycle_summary(IntAls,UpdIs,Cyc) % no: process next cycle
           ;  true
           ),
           NextCyc is Cyc + 1,
           process_cycles(Als,Cs,UpdIs,NextCyc)
        ).

% Process a single cycle ...
process_cycle(Als,Cs,Is,IntAls,UpdIs) :-
        normalize(Is,Cs,NormIs),        % step 1: normalize input nodes
        integrate(Als,NormIs,IntAls),   % step 2: integrate constraints
        feed_back(NormIs,IntAls,UpdIs). % step 3: update constraints

threshold(Cyc,T) :-
        delta(Delta),
        T is 1.0 - Cyc * Delta.

threshold_reached(IntAls,Cyc) :-
        threshold(Cyc,T),
        member((_,IntAc),IntAls),
        IntAc >= T.

format_cycle_summary(IntAls,UpdIs,Cyc) :-
        threshold(Cyc,T),
        format('~n'),
        format('%%%% Model state after: ~d processing cycle(s)~n',Cyc),
        format('%%%%~n'),
        format('%%%% Threshold: ~3f~n',T),
        format('%%%%~n'),
        foreach(member((Al,IntAct),IntAls),
          format('%%%% Alternative [~a]: ~3f~n',[Al,IntAct])),
        format('%%%%~n'),
        foreach(member((C,Al,W,Ac),UpdIs),
          format('%%%% Input node [cst: ~a] [alt: ~a] [wgt: ~3f]: ~3f~n',[C,Al,W,Ac])).

format_processing_summary(IntAls,Cyc) :-
        threshold(Cyc,T),
        member((Al,IntAc),IntAls),
        IntAc >= T, !,
        format('~n'),
        format('%%%% Threshold [~3f] reached after: ~d processing cycle(s)~n',[T,Cyc]),
        format('%%%%~n'),
        format('%%%% Winner activation [~a]: ~3f~n',[Al,IntAc]),
        format('~n').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Step 1: Normalization %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

normalize(Is,Cs,NormIs) :-
        sum_activations_per_constraint(Cs,Is,CsSums),
        norm_input_nodes(Is,CsSums,NormIs).

sum_activations_per_constraint([],_,[]).
sum_activations_per_constraint([C|Cs],Is,[(C,Sum)|CsSums]) :-
        findall(Ac,member((C,_,_,Ac),Is),Acs),
        sum_list(Acs,Sum),
        sum_activations_per_constraint(Cs,Is,CsSums).

norm_input_nodes([],_,[]).
norm_input_nodes([(C,Al,W,Ac)|Is],CsSums,[(C,Al,W,NormAc)|NormIs]) :-
        memberchk((C,Sum),CsSums),
        NormAc is Ac / Sum,
        norm_input_nodes(Is,CsSums,NormIs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Step 2: Integration %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

integrate([],_,[]).
integrate([Al|Als],Is,[(Al,IntAc)|IntAls]) :-
        findall((W,Ac),member((_,Al,W,Ac),Is),WgAcs),
        integrate_constraints(WgAcs,0,IntAc),
        integrate(Als,Is,IntAls).

integrate_constraints([],IntAc,IntAc).
integrate_constraints([(W,Ac)|Is],IntAcAcc0,IntAc) :-
        IntAcAcc1 is IntAcAcc0 + W * Ac,
        integrate_constraints(Is,IntAcAcc1,IntAc).

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Step 3: Feed back %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

feed_back([],_,[]).
feed_back([(C,Al,W,Ac)|Is],IntAls,[(C,Al,W,UpdAc)|UpdIs]) :-
        memberchk((Al,IntAc),IntAls),
        UpdAc is Ac + IntAc * W * Ac,
        feed_back(Is,IntAls,UpdIs).
