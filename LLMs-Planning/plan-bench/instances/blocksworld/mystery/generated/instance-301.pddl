(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e k c d j)
(:init 
(harmony)
(planet e)
(planet k)
(planet c)
(planet d)
(planet j)
(province e)
(province k)
(province c)
(province d)
(province j)
)
(:goal
(and
(craves e k)
(craves k c)
(craves c d)
(craves d j)
)))