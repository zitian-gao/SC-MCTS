(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b h d a c k l f j g i e)
(:init 
(handempty)
(ontable b)
(ontable h)
(ontable d)
(ontable a)
(ontable c)
(ontable k)
(ontable l)
(ontable f)
(ontable j)
(ontable g)
(ontable i)
(ontable e)
(clear b)
(clear h)
(clear d)
(clear a)
(clear c)
(clear k)
(clear l)
(clear f)
(clear j)
(clear g)
(clear i)
(clear e)
)
(:goal
(and
(on b h)
(on h d)
(on d a)
(on a c)
(on c k)
(on k l)
(on l f)
(on f j)
(on j g)
(on g i)
(on i e)
)))