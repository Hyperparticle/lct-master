:- use_module(cim).

% The delta parameter, used to control the activation threshold.
delta(0.0075).

% Flags whether model state summaries are given after each cycle.
flag_verbose :- false.

%% input_node(-Constraint,-Alternative,-Weight,-Activation).
%
%  This predicate is used to specify the model layout. Each line specifies
%  a Constraint for an Alternative, the weight for this Constraint, and its
%  initial Activation.
%
%  Each constraint should be fully specified for each alternative, and
%  weights should ideally sum to 1.0. Note: this will not be checked,
%  leaving you free to play around with different configurations.

input_node('constraint1','alternative1',1,0.33).
input_node('constraint1','alternative2',1,0.66).

input_node('constraint2','alternative1',0,0.33).
input_node('constraint2','alternative2',0,0.66).
