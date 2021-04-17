extensions [ py ]

breed [mice mouse]
breed [hawks hawk]
globals [

  generation
  mice-avg-dfs
  hawks-avg-dfs
  mice-avg-lrs
  hawks-avg-lrs
  mice-avg-batch
  hawks-avg-batch
  mice-avg-lyrs
  hawks-avg-lyrs
  mice-avg-pls
  hawks-avg-pls
  mice-avg-sls
  hawks-avg-sls
  mice-avg-eds
  hawks-avg-eds
  mice-avg-scrs
  hawks-avg-scrs
  mice-avg-sscrs
  hawks-avg-sscrs
]
turtles-own [reward last-state action score scaled-score params]
mice-own [ dead? ]

to setup
  clear-all
  set generation 0
  set mice-avg-dfs []
  set hawks-avg-dfs []
  set mice-avg-lrs []
  set hawks-avg-lrs []
  set mice-avg-batch []
  set hawks-avg-batch []
  set mice-avg-lyrs []
  set hawks-avg-lyrs []
  set mice-avg-pls []
  set hawks-avg-pls []
  set mice-avg-sls []
  set hawks-avg-sls []
  set mice-avg-eds []
  set hawks-avg-eds []
  set mice-avg-scrs []
  set hawks-avg-scrs []
  set mice-avg-sscrs []
  set hawks-avg-sscrs []

  setup-python-environment

  ask patches [set pcolor 52]

  if walls-on? [
    ask patches with [abs pxcor = 12 and abs pycor >= 8 and abs pycor <= 12] [set pcolor white]
    ask patches with [abs pycor = 12 and abs pxcor >= 8 and abs pxcor <= 12] [set pcolor white]
    ask patches with [pxcor = 0 and abs pycor <= 8 and abs pycor >= 4][set pcolor white]
    ask patches with [pycor = 0 and abs pxcor <= 8 and abs pxcor >= 4][set pcolor white]
  ]

  create-hawks 10 [
    set size 2
    set color one-of [blue red yellow cyan]
    let targ one-of patches with [pcolor = 52 and not any? hawks-here]
    set xcor [pxcor] of targ
    set ycor [pycor] of targ
    set params new-params
    set reward 0
    set action 0
    set last-state n-values 36 [0]
    set shape "hawk"
    py:set "params" format-params params
    py:set "id" who
    py:run "agents[id] = q.AgentNormalBatch(params[0],0.5,params[1],36,params[2],3, layers=params[3], fc1_dim=params[4], fc2_dim=params[5], eps_dec = params[6])"
  ]
  create-mice 10 [
    set size 1.5
    set color one-of [white grey brown black]
    let targ one-of patches with [pcolor = 52 and not any? hawks-here and not any? mice-here]
    set xcor [pxcor] of targ
    set ycor [pycor] of targ
    set params new-params
    set reward 0
    set action 0
    set last-state n-values 36 [0]
    set shape "mouse"
    set dead? false
    py:set "params" format-params params
    py:set "id" who
    py:run "agents[id] = q.AgentNormalBatch(params[0],0.5,params[1],36,params[2],3, layers=params[3], fc1_dim=params[4], fc2_dim=params[5], eps_dec = params[6])"
  ]
  reset-ticks
end

to go
  if ticks != 0 and ((ticks mod gen-length) = 0) and generational-evolution? [make-next-gen]
  ask hawks [mice-act]
  ask mice [mice-act]
  tick
end

to-report observations
  let result []
  repeat 12 [
    set result sentence result observe-cone 30 7
    lt 30
  ]
  report map [i -> i / 7.0] result
end

to-report observe-cone [angle depth]
  let result []
  let wall min-one-of patches in-cone depth angle with [pcolor = white][distance myself]
  let min-dist ifelse-value wall = nobody [depth][distance wall]
  let near other turtles in-cone depth angle
  set result lput (ifelse-value min-dist < depth [depth - min-dist / 2][0.0]) result
  let h min-one-of near with [is-hawk? self] [distance myself]
  (if-else
    h = nobody
    [set result lput 0.0 result]
    [distance myself] of h > min-dist
    [set result lput 0.0 result]
    [set result lput (depth - (([distance myself] of h) / 2)) result]
  )
  let m min-one-of near with [is-mouse? self] [distance myself]
  (if-else
    m = nobody
    [set result lput 0.0 result]
    [distance myself] of m > min-dist
    [set result lput 0.0 result]
    [set result lput (depth - (([distance myself] of m) / 2)) result]
  )
  report result
end

to mice-act
  let state observations
  py:set "id" who
  py:set "state" last-state
  py:set "new_state" state
  py:set "reward" reward
  py:set "terminal" 1
  py:set "action" action
  py:run "agents[id].store_transition(state, action, reward, new_state, terminal)"
  py:run "agents[id].learn()"
  set last-state state
  py:set "obs" state
  set action py:runresult "agents[id].choose_action(obs)"
  (ifelse
    action = 0
    [lt 15]
    action = 1
    [rt 15]
    action = 2
    [if [pcolor = 52] of patch-ahead 0.4 [fd 0.4]]
  )
  do-rewards
  ;show word "S:" state
  ;show word "S':" observations
  ;ask other turtles in-cone 7 60 [hatch-vis 1 [set color blue set shape "dot"]]
end

to setup-python-environment
  py:setup py:python
  py:run "import qlearnnlogo as q"
  py:run "import pickle"
  py:run "agents = dict()"
end

to do-rewards
  if-else is-hawk? self [
    set reward 0
    if any? mice in-radius 0.5 with [not dead?] [
      set reward 1
      set score score + 1
      ask one-of mice in-radius 0.5 with [not dead?] [set dead? true]
    ]
  ]
  [
    ifelse dead? [
      set reward -1
      set score score - 1
      move-to one-of patches with [pcolor = 52 and not any? hawks-here and not any? mice-here]
      set dead? false
    ]
    [
      set reward 0
    ]
  ]
end

to-report new-params
  let result [] ; 1. Discount factor 0.9999: * 2 ^ (0 - 9) Learning rate: 0.002 ^ (1 /(0.85 - 1.5)) Batch: 2^(5 6 7 8), Layers: (2 - 5), FC1: 36 - 128, FC2: 16 - 128, eps_dec: 0.99995 ^ 2 ^ (0 - 10)
  set result lput random-float 10 result
  set result lput ((random-float 0.65) + 0.85) result
  set result lput ((random-float 3) + 5) result
  set result lput ((random 4) + 2) result
  set result lput ((random 93) + 36) result
  set result lput ((random 113) + 16) result
  set result lput (random-float 10) result
  report result
end

to-report format-params [p]
  let results []
  set results lput (0.9999 ^ (2 ^ (item 0 p))) results
  set results lput (0.002 ^ (1 / (item 1 p))) results
  set results lput (round (2 ^ (item 2 p))) results
  set results lput (item 3 p) results
  set results lput (item 4 p) results
  set results lput (item 5 p) results
  set results lput (0.99995 ^ (2 ^ (item 6 p))) results
  report results
end

to-report mutate [p]
  let result []
  set result lput min list 10 (max list ((item 0 p) + random-normal 0 0.2) 0) result
  set result lput min list 1.5 (max list ((item 1 p) + random-normal 0 0.05) 0.85) result
  set result lput min list 8 (max list ((item 2 p) + random-normal 0 0.1) 5) result
  set result lput min list 5 (max list (item 3 p + ifelse-value (random-float 1 < 0.3) [0][(random 3) - 1]) 2) result
  set result lput min list 128 (max list (item 4 p + ifelse-value (random-float 1 < 0.3) [0][(random 9) - 4]) 36) result
  set result lput min list 128 (max list (item 5 p + ifelse-value (random-float 1 < 0.3) [0][(random 9) - 4]) 16) result
  set result lput min list 10 (max list ((item 0 p) + random-normal 0 0.2) 0) result
  report result
end

to scale-scores
  let score-range max [score] of hawks - min [score] of hawks
  let score-floor min [score] of hawks
  ask hawks [
    set scaled-score ((score - score-floor) + (0.2 * score-range)) / (1.2 * score-range)
    py:set "id" who
    let net-size py:runresult "agents[id].Q_eval.model_size()" +  ((item 2 (format-params params)) * 100)
    let ratio (1 - (net-size / 80259))
    set scaled-score scaled-score * ratio
  ]

  set score-range max [score] of mice - min [score] of mice
  set score-floor min [score] of mice
  ask mice  [
    set scaled-score ((score - score-floor) + (0.2 * score-range)) / (1.2 * score-range)
    py:set "id" who
    let net-size py:runresult "agents[id].Q_eval.model_size()" +  ((item 2 (format-params params)) * 100)
    let ratio (1 - (net-size / 80259))
    set scaled-score scaled-score * ratio
  ]
end

to make-next-gen

  scale-scores

  show-stats
  do-plotting
  set generation generation + 1

  let hawk-params map [i -> [params] of i] sort-on [scaled-score] max-n-of 4 hawks [scaled-score]
  let mice-params map [i -> [params] of i] sort-on [scaled-score] max-n-of 4 mice [scaled-score]
  ask turtles [set params false]

  ask n-of 4 mice with [params = false][reset-turtle mutate item 3 mice-params]
  ask n-of 3 mice with [params = false][reset-turtle mutate item 2 mice-params]
  ask n-of 2 mice with [params = false][reset-turtle mutate item 1 mice-params]
  ask n-of 1 mice with [params = false][reset-turtle mutate item 0 mice-params]

  ask n-of 4 hawks with [params = false][reset-turtle mutate item 3 hawk-params]
  ask n-of 3 hawks with [params = false][reset-turtle mutate item 2 hawk-params]
  ask n-of 2 hawks with [params = false][reset-turtle mutate item 1 hawk-params]
  ask n-of 1 hawks with [params = false][reset-turtle mutate item 0 hawk-params]
end

to show-stats
  output-print (sentence "Generation:" generation)
  foreach range (count turtles) [x -> ask turtle x [output-show (sentence score scaled-score (format-params params))]]
end

to do-plotting

  let mice-avg-df mean [item 0 format-params params] of mice
  let hawks-avg-df mean [item 0 format-params params] of hawks
  let mice-avg-lr mean [item 1 format-params params] of mice
  let hawks-avg-lr mean [item 1 format-params params] of hawks
  let mice-avg-bt mean [item 2 format-params params] of mice
  let hawks-avg-bt mean [item 2 format-params params] of hawks
  let mice-avg-lyr mean [item 3 format-params params] of mice
  let hawks-avg-lyr mean [item 3 format-params params] of hawks
  let mice-avg-pl mean [item 4 format-params params] of mice
  let hawks-avg-pl mean [item 4 format-params params] of hawks
  let mice-avg-sl mean [item 5 format-params params] of mice
  let hawks-avg-sl mean [item 5 format-params params] of hawks
  let mice-avg-ep mean [item 6 format-params params] of mice
  let hawks-avg-ep mean [item 6 format-params params] of hawks
  let mice-avg-scr mean [score] of mice
  let hawks-avg-scr mean [score] of hawks
  let mice-avg-sscr mean [scaled-score] of mice
  let hawks-avg-sscr mean [scaled-score] of hawks

  set mice-avg-dfs lput mice-avg-df mice-avg-dfs
  set hawks-avg-dfs lput hawks-avg-df hawks-avg-dfs
  set mice-avg-lrs lput mice-avg-lr mice-avg-lrs
  set hawks-avg-lrs lput hawks-avg-lr hawks-avg-lrs
  set mice-avg-batch lput mice-avg-bt mice-avg-batch
  set hawks-avg-batch lput hawks-avg-bt hawks-avg-batch
  set mice-avg-lyrs lput mice-avg-lyr mice-avg-lyrs
  set hawks-avg-lyrs lput hawks-avg-lyr hawks-avg-lyrs
  set mice-avg-pls lput mice-avg-pl mice-avg-pls
  set hawks-avg-pls lput hawks-avg-pl hawks-avg-pls
  set mice-avg-sls lput mice-avg-sl mice-avg-sls
  set hawks-avg-sls lput hawks-avg-sl hawks-avg-sls
  set mice-avg-eds lput mice-avg-ep mice-avg-eds
  set hawks-avg-eds lput hawks-avg-ep hawks-avg-eds
  set mice-avg-scrs lput mice-avg-scr mice-avg-scrs
  set hawks-avg-scrs lput hawks-avg-scr hawks-avg-scrs
  set mice-avg-sscrs lput mice-avg-sscr mice-avg-sscrs
  set hawks-avg-sscrs lput hawks-avg-sscr hawks-avg-sscrs

  set-current-plot "Average Discount Factor"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-df
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-df
  set-current-plot "Average Learning Rate"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-lr
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-lr
  set-current-plot "Average Batch Size"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-bt
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-bt
  set-current-plot "Average Layers"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-lyr
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-lyr
  set-current-plot "Average Primary Layer Dimension"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-pl
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-pl
  set-current-plot "Average Secondary Layer Dimension"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-sl
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-sl
  set-current-plot "Average Epsilon Decrement"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-ep
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-ep
  set-current-plot "Average Scores"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-scr
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-scr
  set-current-plot "Average Scaled Scores"
  set-current-plot-pen "Hawks"
  plotxy generation hawks-avg-sscr
  set-current-plot-pen "Mice"
  plotxy generation mice-avg-sscr

end

to reset-turtle [p]
  set params p
  set reward 0
  set action 0
  set score 0
  set last-state n-values 36 [0]
  if is-mouse? self [set dead? false]
  py:set "params" format-params params
  py:set "id" who
  py:run "agents[id] = q.AgentNormalBatch(params[0],0.5,params[1],36,params[2],3, layers=params[3], fc1_dim=params[4], fc2_dim=params[5], eps_dec = params[6])"
end
@#$#@#$#@
GRAPHICS-WINDOW
210
10
878
679
-1
-1
20.0
1
10
1
1
1
0
1
1
1
-16
16
-16
16
1
1
1
ticks
30.0

BUTTON
17
15
80
48
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
17
50
80
83
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
16
186
79
219
save
py:set \"fname\" f-name\n(py:run\n\"with open(fname, 'wb') as filehandler:\"\n\"   pickle.dump(agents, filehandler)\"\n )
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
85
186
148
219
load
py:set \"fname\" f-name\n(py:run\n\"with open(fname, 'rb') as filehandler:\"\n\"   agents = pickle.load(filehandler)\"\n )
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

OUTPUT
882
11
1535
312
11

PLOT
1087
316
1287
466
Average Learning Rate
Generation
Learning Rate
0.0
10.0
6.0E-4
0.016
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
882
316
1082
466
Average Discount Factor
Generation
Discount Factor
0.0
10.0
0.9
1.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
1291
316
1491
466
Average Batch Size
Generation
Batch Size
0.0
10.0
32.0
256.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
882
470
1082
620
Average Layers
Generation
Layers
0.0
10.0
2.0
5.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
1087
471
1287
621
Average Primary Layer Dimension
Generation
Dimension
0.0
10.0
36.0
128.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
1291
471
1491
621
Average Secondary Layer Dimension
Generation
Dimension
0.0
10.0
16.0
128.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

INPUTBOX
16
124
148
184
f-name
agent_brains.p
1
0
String

PLOT
1087
625
1287
775
Average Epsilon Decrement
Generation
Epsilon
0.0
10.0
0.95
1.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

PLOT
882
625
1082
775
Average Scores
Generation
Score
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

SLIDER
15
228
187
261
gen-length
gen-length
1000
100000
100000.0
1000
1
NIL
HORIZONTAL

TEXTBOX
806
700
877
727
Hawks
22
105.0
1

TEXTBOX
815
725
870
752
Mice
22
15.0
1

PLOT
1291
626
1491
776
Average Scaled Scores
Generation
Score
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"Hawks" 1.0 0 -13345367 true "" ""
"Mice" 1.0 0 -2674135 true "" ""

SWITCH
11
265
197
298
generational-evolution?
generational-evolution?
1
1
-1000

SWITCH
88
36
194
69
walls-on?
walls-on?
1
1
-1000

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

cat
true
0
Polygon -7500403 true true 16 150 16 135 46 120 52 93 76 120 91 120 106 105 151 105 166 120 196 105 241 105 256 120 256 135 286 150 299 168 283 209 242 233 226 225 271 195 279 170 256 165 256 180 241 195 196 195 166 180 151 195 106 195 91 180 76 180 49 209 46 180 16 165
Circle -13345367 true false 30 154 12
Circle -13345367 true false 30 131 12
Polygon -2064490 true false 17 155 18 143 24 151
Line -16777216 false 21 161 17 191
Line -16777216 false 21 139 17 109
Line -16777216 false 28 137 24 107
Line -16777216 false 28 163 24 193
Polygon -2064490 true false 49 180 59 194 52 205
Polygon -2064490 true false 49 120 59 106 52 95

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

hawk
true
0
Polygon -7500403 true true 150 15 135 45 120 60 135 75 30 105 0 165 135 120 120 195 180 195 165 120 300 165 270 105 165 75 180 60 165 45 150 15
Polygon -7500403 true true 30 165 45 135 135 120 135 135
Polygon -7500403 true true 270 165 255 135 165 120 165 135
Polygon -7500403 true true 255 180 240 150 165 120 165 150
Polygon -7500403 true true 45 180 60 150 135 120 135 150
Polygon -7500403 true true 165 150 195 225 150 195
Polygon -7500403 true true 135 150 105 225 150 195
Polygon -7500403 true true 135 150 120 240 165 225
Polygon -7500403 true true 165 165 180 240 135 225
Polygon -2674135 true false 165 45 150 30 150 15 165 45 150 30
Polygon -2674135 true false 135 45 150 30 150 15 135 45 150 30
Polygon -13791810 true false 165 45 165 60 165 60 150 45 165 45
Polygon -13791810 true false 135 45 135 60 135 60 150 45 135 45

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

mouse
true
0
Polygon -7500403 true true 135 225 120 210 105 180 105 150 105 150 105 150 105 135 120 90 85 88 105 60 120 75 150 15 180 75 195 60 209 88 180 90 195 135 195 150 195 150 195 180 180 210 165 225
Circle -16777216 true false 134 53 12
Circle -16777216 true false 154 53 12
Line -7500403 true 150 45 120 60
Line -7500403 true 150 45 120 45
Line -7500403 true 150 45 120 30
Line -7500403 true 150 45 180 45
Line -7500403 true 150 45 180 30
Line -7500403 true 150 45 180 60
Circle -2064490 true false 144 12 12
Line -7500403 true 150 225 135 240
Line -7500403 true 135 240 135 255
Line -7500403 true 135 255 165 270
Line -7500403 true 165 270 180 255
Line -7500403 true 180 255 165 240
Polygon -7500403 true true 120 210 90 165 105 165 120 195
Polygon -7500403 true true 180 210 210 165 195 165 180 195
Polygon -7500403 true true 165 135 180 105 195 105 180 135
Polygon -7500403 true true 135 135 120 105 105 105 120 135
Polygon -2064490 true false 118 86 94 81 104 71
Polygon -2064490 true false 178 86 202 81 192 71

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
