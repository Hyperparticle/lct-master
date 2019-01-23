:- use_module(cim).

% The delta parameter, used to control the activation threshold.
delta(0.0075).

% Flags whether model state summaries are given after each cycle.
flag_verbose :- true.

%% input_node(-Constraint,-Alternative,-Weight,-Activation).
%
%  This predicate is used to specify the model layout. Each line specifies
%  a Constraint for an Alternative, the weight for this Constraint, and its
%  initial Activation.
%
%  Each constraint should be fully specified for each alternative, and
%  weights should ideally sum to 1.0. Note: this will not be checked,
%  leaving you free to play around with different configurations.
input_node('constraint1','alternative1',0.50,7).
input_node('constraint1','alternative2',0.50,4).

input_node('constraint2','alternative1',0.25,0.10).
input_node('constraint2','alternative2',0.25,0.90).

input_node('constraint3','alternative1',0.10,0.75).
input_node('constraint3','alternative2',0.10,0.25).

input_node('constraint4','alternative1',0.15,0.20).
input_node('constraint4','alternative2',0.15,0.30).
