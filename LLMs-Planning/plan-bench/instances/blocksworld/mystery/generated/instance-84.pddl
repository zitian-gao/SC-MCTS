(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f k l e g h d b j c i a)
(:init 
(harmony)
(planet f)
(planet k)
(planet l)
(planet e)
(planet g)
(planet h)
(planet d)
(planet b)
(planet j)
(planet c)
(planet i)
(planet a)
(province f)
(province k)
(province l)
(province e)
(province g)
(province h)
(province d)
(province b)
(province j)
(province c)
(province i)
(province a)
)
(:goal
(and
(craves f k)
(craves k l)
(craves l e)
(craves e g)
(craves g h)
(craves h d)
(craves d b)
(craves b j)
(craves j c)
(craves c i)
(craves i a)
)))