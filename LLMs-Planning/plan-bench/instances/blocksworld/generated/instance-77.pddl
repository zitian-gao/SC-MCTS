(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i g f e a k c l b)
(:init 
(handempty)
(ontable i)
(ontable g)
(ontable f)
(ontable e)
(ontable a)
(ontable k)
(ontable c)
(ontable l)
(ontable b)
(clear i)
(clear g)
(clear f)
(clear e)
(clear a)
(clear k)
(clear c)
(clear l)
(clear b)
)
(:goal
(and
(on i g)
(on g f)
(on f e)
(on e a)
(on a k)
(on k c)
(on c l)
(on l b)
)))