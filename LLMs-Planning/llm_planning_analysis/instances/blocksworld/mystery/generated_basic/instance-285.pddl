

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(planet b)
(craves c b)
(planet d)
(province a)
(province c)
)
(:goal
(and
(craves a b)
(craves c d)
(craves d a))
)
)


