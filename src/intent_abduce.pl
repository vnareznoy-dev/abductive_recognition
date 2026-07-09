% intent_abduce.pl
% ---------------------------------------------------------------------------
% Abductive symbolic classifier of an evader's INTENT (System 2).
%
% This is the white-box counterpart to the inductive LSTM/GRU baselines.
% There is NO training and NO history model: given the current
% frame's geometric observation o, we search for the minimal intent assumption
% Delta that, together with the background theory T, ENTAILS the observation
% and stays consistent with the integrity constraints IC:
%
%     Delta* = argmin |Delta|  s.t.  T u Delta |= observed(o)
%                                     T u Delta u IC  not|- false
%
% Here every Delta is a single intent atom (|Delta| = 1), so minimality reduces
% to: among the intents whose rule body the observation satisfies, choose the
% most specific one that violates no integrity constraint. The chosen rule body
% IS the proof / justification (white-box) — see classify/2 returning Why.
%
% Thresholds below are DOMAIN KNOWLEDGE (flight geometry), part of T — they are
% authored, not learned. Zero training trajectories are consumed.
% ---------------------------------------------------------------------------

:- dynamic f_speed/1.     % f_speed(V)        evader speed (m/s)
:- dynamic f_yaw_rate/1.  % f_yaw_rate(P)     yaw rate (rad/s)
:- dynamic f_off_axis/1.  % f_off_axis(M)     off-axis miss of goal (m)
:- dynamic f_rgoal/1.     % f_rgoal(R)        range to goal (m)
:- dynamic f_rdot/1.      % f_rdot(Rd)        closing rate on goal (m/s, <0 closing)
:- dynamic f_osc/1.       % f_osc(Bool)       serpentine flag (true/false)

% --- Domain thresholds (background theory T) --------------------------------
thr(off_axis_low, 8.0).   % m below this => velocity points essentially at goal
thr(closing,     -1.5).   % rdot below this => genuinely closing on the goal
thr(hold_speed,   4.0).   % speed below this => not committing to a dash
thr(hold_rdot,   -1.5).   % rdot above this => not closing (loitering)

% --- Background theory T: intent signatures ---------------------------------
% Each clause is a logical explanation: "IF the geometry looks like this, the
% intent `rush/flank/weave/hold` ENTAILS that observation." Bodies are crisp
% and human-auditable.

intent_rule(weave, osc(serpentine)) :-
    f_osc(true).

intent_rule(hold, loiter(slow, not_closing)) :-
    f_speed(V), thr(hold_speed, VS), V < VS,
    f_rdot(Rd), thr(hold_rdot, RS), Rd >= RS.

intent_rule(rush, aimed_at_goal(low_miss, closing)) :-
    f_osc(false),
    f_off_axis(M), thr(off_axis_low, ML), M < ML,
    f_rdot(Rd), thr(closing, RC), Rd < RC.

intent_rule(flank, offset_approach(high_miss, closing)) :-
    f_osc(false),
    f_off_axis(M), thr(off_axis_low, ML), M >= ML,
    f_rdot(Rd), thr(closing, RC), Rd < RC.

% --- Integrity constraints IC -----------------------------------------------
% A consistent intent assumption must not be simultaneously explainable as a
% contradictory one. Serpentine motion forbids the committed-dash readings;
% a genuine loiter forbids a closing dash. These prune incoherent abductions.
incompatible(weave, rush).
incompatible(weave, flank).
incompatible(hold,  rush).
incompatible(hold,  flank).

violates_ic(I) :-
    intent_rule(I, _),
    incompatible(I, J),
    intent_rule(J, _),       % some contradictory intent also fires
    !.

% --- Specificity ordering (drives minimal, decisive explanation) ------------
% Lower number = more diagnostic / more specific signature, tried first.
specificity(weave, 0).
specificity(hold,  1).
specificity(flank, 2).
specificity(rush,  3).

% --- Abduction --------------------------------------------------------------
% candidate intents = those whose rule fires AND violate no IC, ordered by
% specificity; the head of that list is the minimal best explanation.
classify(Intent, Why) :-
    findall(S-(I-W),
            ( intent_rule(I, W), \+ violates_ic(I), specificity(I, S) ),
            L),
    L \= [],
    sort(L, [_-(Intent-Why) | _]).

% Fallback when no rule fires (degenerate frame): default to the least
% committal explanation, flagged as such, so the caller still gets an answer.
classify(hold, default(no_rule_fired)) :-
    \+ ( intent_rule(I, _), \+ violates_ic(I) ).

% Convenience: intent only.
classify(Intent) :- classify(Intent, _).
