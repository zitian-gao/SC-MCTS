(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i f e l g k j b)
(:init 
(harmony)
(planet i)
(planet f)
(planet e)
(planet l)
(planet g)
(planet k)
(planet j)
(planet b)
(province i)
(province f)
(province e)
(province l)
(province g)
(province k)
(province j)
(province b)
)
(:goal
(and
(craves i f)
(craves f e)
(craves e l)
(craves l g)
(craves g k)
(craves k j)
(craves j b)
)))