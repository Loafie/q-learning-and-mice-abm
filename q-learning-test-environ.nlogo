extensions [ py ]

breed [mice mouse]
breed [fruit a-fruit]
breed [poison a-poison]
breed [vis a-vis]

mice-own[fruits poisons last-state reward action term reward-type fruit-ratio score ratios]

to setup
  clear-all
  setup-python-environment
  ask patches [set pcolor 51]
  ask n-of num-fruit patches [
    sprout-fruit 1 [
      set color red
      set shape "fruit"
    ]
  ]
  ask n-of num-poison patches with [count fruit-here = 0] [
    sprout-poison 1 [
      set color magenta
      set shape "poison"
    ]
  ]
  create-mice 1 [
    set color white
    set shape "mouse"
    set size 2
    set last-state [0 0 0 0 0 0 0 0 0]
    set fruits 1
    set poisons 1
    set reward-type 3
    set reward 0
    set action 0
    set ratios [0.5]
    py:set "id" who
    py:run "agents[id] = q.AgentNBND(0.9995,0.5,0.002,9,500,3, layers=3, fc1_dim=256, fc2_dim=256, eps_dec = 0.99996)"
  ]
;  create-mice 1 [
;    set color brown
;    set shape "mouse"
;    set size 2
;    set last-state [0 0 0 0 0 0 0 0 0 0 0 0]
;    set fruits 1
;    set poisons 1
;    set reward-type 1
;    set reward 0
;    set action 0
;    py:set "id" who
;    py:run "agents[id] = q.AgentBatchRewardProportional(0.99,0.5,0.002,12,64,3, layers=3, fc1_dim=32, fc2_dim=32, eps_dec = 0.99996)"
;
;  ]
;  create-mice 1 [
;    set color grey
;    set shape "mouse"
;    set size 2
;    set last-state [0 0 0 0 0 0 0 0 0]
;    set fruits 1
;    set poisons 1
;    set reward-type 3
;    set reward 0
;    set action 0
;    py:set "id" who
;    py:run "agents[id] = q.AgentNormalBatch(0.99,0.5,0.002,9,64,3, layers=2, fc1_dim=12, fc2_dim=12)"
;
;  ]
;  create-mice 1 [
;    set color black
;    set shape "mouse"
;    set fruits 1
;    set poisons 1
;    set size 2
;    set last-state [0 0 0 0 0 0 0 0 0]
;    set reward 0
;    set action 0
;    py:set "id" who
;    py:run "agents[id] = q.Agent(0.99,0.7,0.01,9,64,3,fc1_dim=16,fc2_dim=16)"
;  ]
  reset-ticks
end

to go
  resupply
  ask vis [die]
  ask mice [
    mice-act
    if ticks mod 1000 = 0 [calculate-ratio]
  ]
  tick
end

to calculate-ratio
  if fruits = 0 [set fruits 1]
  if poisons = 0 [set poisons 1]
  set ratios lput (fruits / (fruits + poisons)) ratios
  if length ratios > 10 [set ratios but-first ratios]
  set fruits 0
  set poisons 0
end

to mice-act
  let state observations
  py:set "id" who
  py:set "state" last-state
  py:set "new_state" state
  py:set "reward" reward
  py:set "terminal" term
  py:set "action" action
  py:run "agents[id].store_transition(state, action, reward, new_state, terminal)"
  py:run "agents[id].learn()"
  set last-state state
  py:set "obs" state
  set action py:runresult "agents[id].choose_action(obs)"
  (ifelse
    action = 0
    [lt 20]
    action = 1
    [rt 20]
    action = 2
    [fd 0.4]
  )
  do-rewards
  ;show word "S:" state
  ;show word "S':" observations
  ;ask other turtles in-cone 7 60 [hatch-vis 1 [set color blue set shape "dot"]]
end

to resupply
  if count poison < num-poison [
    ask n-of (num-poison - count poison) patches with [count poison-here = 0 and count fruit-here = 0] [
      sprout-poison 1 [
        set color magenta
        set shape "poison"
      ]
    ]
  ]
  if count fruit < num-fruit [
    ask n-of (num-fruit - count fruit) patches with [count fruit-here = 0 and count poison-here = 0] [
      sprout-fruit 1 [
        set color red
        set shape "fruit"
      ]
    ]
  ]
end

to-report observations
  let obs []
  let near other turtles in-cone 7 60
  rt 20
  let m min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else m = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance m) / 2)) obs
  ]
  let p min-one-of near with [is-a-poison? self] in-cone 7 20 [distance myself]
  if-else p = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance p) / 2)) obs
  ]
  let f min-one-of near with [is-a-fruit? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  lt 20
  set m min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else m = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance m) / 2)) obs
  ]
  set p min-one-of near with [is-a-poison? self] in-cone 7 20 [distance myself]
  if-else p = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance p) / 2)) obs
  ]
  set f min-one-of near with [is-a-fruit? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  lt 20
  set m min-one-of near with [is-mouse? self] in-cone 7 20 [distance myself]
  if-else m = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance m) / 2)) obs
  ]
  set p min-one-of near with [is-a-poison? self] in-cone 7 20 [distance myself]
  if-else p = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance p) / 2)) obs
  ]
  set f min-one-of near with [is-a-fruit? self] in-cone 7 20 [distance myself]
  if-else f = nobody [
    set obs lput 0 obs
  ][
    set obs lput (7 - ((distance f) / 2)) obs
  ]
  rt 20
;  set obs map [i -> i / 7] obs
;  (ifelse
;    action = 0
;    [set obs sentence obs [1.0 0 0] ]
;    action = 1
;    [set obs sentence obs [0 1.0 0] ]
;    action = 2
;    [set obs sentence obs [0 0 1.0] ])
  report obs
end


to setup-python-environment
  py:setup py:python
  py:run "import qlearnnlogo as q"
  py:run "agents = dict()"
end


to do-rewards
  set term 1
  (if-else
    reward-type = 1
    [
      set reward 0
    ]
    reward-type = 2
    [
      set reward 1
    ]
    reward-type = 3
    [
      if any? (poison in-cone 7 60) [
        let nearest-poison min-one-of (poison in-cone 7 60) [distance myself]
        let p-d distance nearest-poison
        set reward max (list (-5) (- (1 / p-d)))
      ]
      if any? (fruit in-cone 7 60) [
        let nearest-fruit min-one-of (fruit in-cone 7 60) [distance myself]
        let f-d distance nearest-fruit
        set reward min (list (5)  (1 / f-d))
      ]
    ]
    reward-type = 4
    [
      set reward -1
    ]
  )
  let near-poison poison in-radius 0.5
  if any? near-poison [
    ask one-of near-poison [
      die
    ]
    set poisons poisons + 1
    set score score - 1
    (if-else
      reward-type = 1 or reward-type = 3
      [
        set reward -5
      ]
      reward-type = 2
      [
        set reward -1
        set term 0
      ]
      reward-type = 4
      [
        set reward -50
      ]
    )
  ]
  let near-fruit fruit in-radius 0.5
  if any? near-fruit [
    ask one-of near-fruit [
      die
    ]
    set fruits fruits + 1
    set score score + 1
    (if-else
      reward-type = 1 or reward-type = 3
      [
        set reward 5
      ]
      reward-type = 2
      [
        set reward 50
      ]
      reward-type = 4
      [
        set reward 10
      ]
    )
  ]

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
130
15
193
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

SLIDER
23
52
195
85
num-fruit
num-fruit
0
100
100.0
1
1
NIL
HORIZONTAL

SLIDER
23
90
195
123
num-poison
num-poison
0
100
100.0
1
1
NIL
HORIZONTAL

BUTTON
63
15
126
48
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
23
126
93
159
Epsilon
show py:runresult \"agents[200].epsilon\"
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
882
10
1243
345
Fruit Ratio
1K Ticks
Fruit / (Fruit + Posion)
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"Black Mouse" 1.0 0 -16777216 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 3)) [plot [mean ratios] of mouse (num-fruit + num-poison + 3)]"
"Grey Mouse" 1.0 0 -7500403 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 2)) [plot [mean ratios] of mouse (num-fruit + num-poison + 2)]"
"Brown Mouse" 1.0 0 -6459832 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 1)) [plot [mean ratios] of mouse (num-fruit + num-poison + 1)]"
"White Mouse" 1.0 0 -2064490 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison)) [plot [mean ratios] of mouse (num-fruit + num-poison)]"

PLOT
882
348
1243
683
Scores
1K Ticks
Score
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"Black Mouse" 1.0 0 -16777216 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 3)) [plot [score] of mouse (num-fruit + num-poison + 3)]"
"Grey Mouse" 1.0 0 -7500403 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 2)) [plot [score] of mouse (num-fruit + num-poison + 2)]"
"Brown Mouse" 1.0 0 -6459832 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison + 1)) [plot [score] of mouse (num-fruit + num-poison + 1)]"
"White Mouse" 1.0 0 -2064490 true "" "if ticks mod 1000 = 0 and (is-mouse? turtle (num-fruit + num-poison)) [plot [score] of mouse (num-fruit + num-poison)]"

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
Polygon -1184463 true false 81 52 276 142 81 232 51 202 36 157 36 127 51 82 81 52 81 52
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

fruit
false
0
Polygon -7500403 true true 150 90 135 75 75 90 60 120 60 180 105 225 195 225 240 180 240 120 225 90 165 75 150 90
Polygon -10899396 true false 150 90 120 30 150 45 165 15

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

poison
false
0
Polygon -6459832 true false 105 45 195 45 180 75 225 225 75 225 120 75 105 45
Polygon -8630108 true false 135 90 165 90 180 105 180 135 165 165 135 165 120 135 120 105 135 90 135 90
Polygon -6459832 true false 169 110 169 125 154 125 154 110 169 110 169 110
Polygon -6459832 true false 131 110 131 125 146 125 146 110 131 110 131 110
Polygon -8630108 true false 105 210 105 195 195 165 195 180
Polygon -8630108 true false 105 165 105 180 195 210 195 195

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
