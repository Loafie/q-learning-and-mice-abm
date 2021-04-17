breed [cheese a-cheese]
breed [mice mouse]
breed [hawks hawk]

mice-own [energy a-CGP memory sick]
hawks-own [eating]
cheese-own [rancid? age]

to setup
  clear-all
  create-mice starting-mice [
    set shape "mouse"
    set color grey
    set size 2
    set a-CGP generate-random-CGP 21 9 18 20 5 8 2
    set energy starting-energy
    setxy random-xcor random-ycor
    set sick 0
    set memory [0 0 0 0 0 0]
  ]
  create-hawks num-hawks [
    set shape "hawk"
    set color blue
    set size 3
    setxy random-xcor random-ycor
    set eating 0
  ]
  repeat starting-cheese [
    ask one-of patches with [not any? cheese-here] [sprout-cheese 1 [
      set rancid? false
      set shape "cheese"
      set color yellow
      ]
    ]
  ]
  reset-ticks
end

to go
  if random-float 1.0 < cheese-spawn-rate [
    ask up-to-n-of 1 patches with [not any? cheese-here] [sprout-cheese 1 [
      set shape "cheese"
      if-else random-float 1 < rancid-cheese-rate [
        set color green
        set rancid? true
      ]
      [
        set color yellow
        set rancid? false
      ]
      ]
    ]
  ]
  ask hawks [
    if-else eating < 1 [
      if shape != "hawk" [set shape "hawk"]
      let the-mice mice in-radius 10
      if-else any? the-mice [
        face min-one-of the-mice [distance myself]
      ]
      [
        rt random 20
        lt random 20
      ]
      fd 0.3
      if any? mice in-radius 0.5 [
        ask one-of mice in-radius 0.5 [
          die
        ]
        set eating hawks-eat-for-ticks
      ]
    ]
    [
      if shape != "hawk-landed" [set shape "hawk-landed"]
      set eating eating - 1
    ]
  ]
  ask mice [
    let output evaluate a-CGP sentence observations memory
    let d decision sublist output 0 3
    set memory sublist output 3 9
    (if-else d = 0
      [rt 20]
      d = 1
      [fd 0.2]
      d = 2
      [lt 20])
    if any? other mice in-radius 0.5 with [sick > 0] and sick = 0 [
      set sick ticks-to-be-sick
      set color 52
    ]
    if any? cheese in-radius 0.5 [
      let the-cheese one-of cheese in-radius 0.5
      if-else [rancid?] of the-cheese [
        set sick sick + ticks-to-be-sick
        set color 52
      ]
      [
        set energy energy + ifelse-value sick > 0 [cheese-energy-gain / 2.0][cheese-energy-gain]
      ]
      ask the-cheese [die]
    ]
    set energy energy - ifelse-value sick > 0 [2 * energy-per-tick][energy-per-tick]
    if sick > 0 [
      set sick sick - 1
      if sick = 0 [
        set color grey
      ]
    ]
    if energy > reproduce-energy [
      set energy energy / 2.0
      let the-CGP a-CGP
      hatch-mice 1 [
        set a-CGP mutated-CGP (the-CGP) 0.02
        set memory [0 0 0 0 0 0]
      ]
    ]
    if energy < 0 [
      die
    ]
  ]
  ask cheese [
    set age age + 1
    if age > rancid-cheese-decay-ticks and rancid? [ die ]
  ]
  tick
end

to-report decision [l]
  let total sum map abs l
  let current 0
  let roll random-float total
  foreach range length l [x ->
    set current current + abs item x l
    if roll <= current [
      report x
    ]
  ]
end


to-report observations
  let obs []
  let near other turtles in-cone 7 60
  rt 20
  let f min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self and rancid?] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-mouse? self and sick > 0] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-hawk? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  lt 20
  set f min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self and rancid?] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-mouse? self and sick > 0] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-hawk? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  lt 20
  set f min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-a-cheese? self and rancid?] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-mouse? self and sick > 0] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  set f min-one-of near with [is-hawk? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  rt 20
  report obs
end


;; This is all the CGP STUFF:

;; first element of the list is a list of the parameters in the order they are passed to this function
;; additional elements are genes for nodes with [function input0 .... inputn]

to-report mutated-CGP [the-CGP mutation-chance]
  let result (list (first the-CGP))
  let inputs item 0 first the-CGP
  let outputs item 1 first the-CGP
  let rows item 2 first the-CGP
  let columns item 3 first the-CGP
  let levels-back item 4 first the-CGP
  let n-functions item 5 first the-CGP
  let arity item 6 first the-CGP
  foreach (range ((rows * columns) + outputs)) [ x ->
    if-else random-float 1.0 < mutation-chance [
      let the-range input-range (x + inputs) inputs outputs rows columns levels-back
      let the-node []
      if-else (x < (rows * columns)) [
        set the-node lput (random n-functions) the-node
        repeat arity [
          set the-node lput ((random ((item 1 the-range) - (item 0 the-range) + 1)) + item 0 the-range) the-node
        ]
      ]
      [
        set the-node lput ((random ((item 1 the-range) - (item 0 the-range) + 1)) + item 0 the-range) the-node
      ]
      set result lput the-node result

    ]
    [
      set result lput (item (1 + x) the-CGP) result
    ]
  ]
  let a-n active-nodes result
  set result lput (item 0 a-n) result
  let header replace-item 7 (item 0 result) (item 1 a-n)
  set result replace-item 0 result header
  report result
end

to-report generate-random-CGP [inputs outputs rows columns levels-back n-functions arity]
  let result []
  foreach (range ((rows * columns) + outputs)) [ x ->
    let the-range input-range (x + inputs) inputs outputs rows columns levels-back
    let the-node []
    if-else (x < (rows * columns)) [
      set the-node lput (random n-functions) the-node
      repeat arity [
        set the-node lput ((random ((item 1 the-range) - (item 0 the-range) + 1)) + item 0 the-range) the-node
      ]
    ]
    [
      set the-node lput ((random ((item 1 the-range) - (item 0 the-range) + 1)) + item 0 the-range) the-node
    ]
    set result lput the-node result
  ]
  set result fput (list inputs outputs rows columns levels-back n-functions arity) result
  let a-n active-nodes result
  set result lput (item 0 a-n) result
  set result replace-item 0 result (lput (item 1 a-n) (item 0 result))
  report result
end

to-report active-nodes [ the-CGP ]
  let rows (item 2 (item 0 the-CGP))
  let columns (item 3 (item 0 the-CGP))
  let outputs (item 1 (item 0 the-CGP))
  let active-node-list []
  repeat (rows * columns) [
    set active-node-list lput False active-node-list
  ]
  foreach (range outputs) [ x ->
    set active-node-list activate-node active-node-list (item 0 (item (x + 1 + (rows * columns)) the-CGP)) the-CGP
  ]
  let active-count 0
  foreach active-node-list [x -> if x [set active-count active-count + 1]]
  report (list active-node-list active-count)
end

to-report activate-node [ active-node-list the-node the-CGP]
  let inputs (item 0 (item 0 the-CGP))
  let arity (item 6 (item 0 the-CGP))
  if (the-node < inputs) [
    report active-node-list
  ]
  if (item (the-node - inputs) active-node-list)
  [
    report active-node-list
  ]
  let result replace-item (the-node - inputs) active-node-list True
  foreach (range arity) [ x ->
    set result activate-node result (item (x + 1) (item (the-node - inputs + 1) the-CGP)) the-CGP
  ]
  report result
end

to-report input-range [node inputs outputs rows columns levels-back]
  (if-else node < inputs
    [report (list node node)]
    node >= (inputs + (rows * columns))
    [report (list 0 (inputs - 1 + (rows * columns)))]
    [
      let upper ((int ((node - inputs) / rows)) * rows) - 1 + inputs
      let lower (upper - (levels-back * rows) + 1)
      if lower < 0 [set lower 0]
      report (list lower upper)
    ]
  )
end

to-report functions [n]
  (if-else n = 0
    [report [x -> item 0 x + item 1 x]]
    n = 1
    [report [x -> item 0 x - item 1 x]]
    n = 2
    [report [x -> item 0 x * item 1 x]]
    n = 3
    [report [x -> ifelse-value ((item 1 x) != 0) [item 0 x / item 1 x] [0]]]
    n = 4
    [report [x -> 1]]
    n = 5
    [report [x -> ifelse-value item 0 x < item 1 x [1][0]]]
    n = 6
    [report [x -> ifelse-value item 0 x > 0 and item 1 x > 0 [1][0]]]
    n = 7
    [report [x -> ifelse-value item 0 x > 0 or item 1 x > 0 [1][0]]]
  )
  report [x -> 1]
end

to-report evaluate [the-CGP input-list]
  let active-node-list last the-CGP
  let evaluations []
  let inputs item 0 first the-CGP
  let outputs item 1 first the-CGP
  let rows item 2 first the-CGP
  let columns item 3 first the-CGP
  foreach (range length active-node-list) [ x ->
    if-else (item x active-node-list) = True [
      let the-node item (x + 1) the-CGP
      let f-num item 0 the-node
      let the-values []
      foreach (range ((length the-node) - 1)) [y ->
        let ref-node item (y + 1) the-node
        if-else (ref-node < inputs) [
          set the-values lput item ref-node input-list the-values
        ]
        [
          set the-values lput (item (ref-node - inputs) evaluations) the-values
        ]
      ]
      let value 0
      carefully [
        set value item 0 (map (functions f-num) (list the-values))
      ]
      [
        set value item 0 the-values
      ]
      set evaluations lput value evaluations
    ]
    [
      set evaluations lput 0 evaluations
    ]
  ]
  let output []
  foreach (range outputs) [ x ->
    let out-node item 0 (item (1 + (rows * columns) + x) the-CGP)
    if-else (out-node < inputs) [
      set output lput (item out-node input-list) output
    ]
    [
      set output lput (item (out-node - inputs) evaluations) output
    ]
  ]
  report output
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

SLIDER
34
10
206
43
starting-mice
starting-mice
0
100
30.0
1
1
NIL
HORIZONTAL

SLIDER
34
46
206
79
starting-cheese
starting-cheese
0
100
50.0
1
1
NIL
HORIZONTAL

BUTTON
35
83
98
116
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
101
83
164
116
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

SLIDER
34
120
206
153
cheese-spawn-rate
cheese-spawn-rate
0
1
0.15
0.01
1
NIL
HORIZONTAL

SLIDER
34
155
206
188
rancid-cheese-rate
rancid-cheese-rate
0
1
0.15
0.01
1
NIL
HORIZONTAL

SLIDER
34
190
206
223
cheese-energy-gain
cheese-energy-gain
0
100
45.0
1
1
NIL
HORIZONTAL

SLIDER
34
225
206
258
ticks-to-be-sick
ticks-to-be-sick
0
100
100.0
1
1
NIL
HORIZONTAL

SLIDER
34
260
206
293
energy-per-tick
energy-per-tick
0
2
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
34
295
206
328
reproduce-energy
reproduce-energy
0
100
100.0
1
1
NIL
HORIZONTAL

SLIDER
34
330
206
363
starting-energy
starting-energy
0
100
50.0
1
1
NIL
HORIZONTAL

SLIDER
34
365
207
398
rancid-cheese-decay-ticks
rancid-cheese-decay-ticks
0
1000
700.0
1
1
NIL
HORIZONTAL

SLIDER
34
400
207
433
hawks-eat-for-ticks
hawks-eat-for-ticks
0
500
150.0
1
1
NIL
HORIZONTAL

SLIDER
35
436
207
469
num-hawks
num-hawks
0
10
3.0
1
1
NIL
HORIZONTAL

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

cheese
false
0
Polygon -7500403 true true 81 52 276 142 81 232 51 202 36 157 36 127 51 82 81 52 81 52
Circle -16777216 true false 72 81 30
Circle -16777216 true false 107 121 24
Circle -16777216 true false 153 124 40
Circle -16777216 true false 74 175 18
Circle -16777216 true false 106 165 33
Circle -16777216 true false 55 131 29
Circle -16777216 true false 205 126 26
Circle -16777216 true false 134 93 24

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

hawk-landed
true
0
Polygon -7500403 true true 150 15 135 45 120 60 135 75 75 135 135 285 150 240 150 165 150 135 150 240 165 285 225 135 165 75 180 60 165 45 150 15
Polygon -2674135 true false 165 45 150 30 150 15 165 45 150 30
Polygon -2674135 true false 135 45 150 30 150 15 135 45 150 30
Polygon -13791810 true false 165 45 165 60 165 60 150 45 165 45
Polygon -13791810 true false 135 45 135 60 135 60 150 45 135 45
Polygon -7500403 true true 210 135 210 210 165 240
Polygon -7500403 true true 90 135 90 210 135 240

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
NetLogo 6.2.0
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
