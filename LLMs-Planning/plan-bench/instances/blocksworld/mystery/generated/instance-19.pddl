(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l j b k e d i g h f)
(:init 
(harmony)
(planet l)
(planet j)
(planet b)
(planet k)
(planet e)
(planet d)
(planet i)
(planet g)
(planet h)
(planet f)
(province l)
(province j)
(province b)
(province k)
(province e)
(province d)
(province i)
(province g)
(province h)
(province f)
)
(:goal
(and
(craves l j)
(craves j b)
(craves b k)
(craves k e)
(craves e d)
(craves d i)
(craves i g)
(craves g h)
(craves h f)
)))