(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a b c j f d e k l g i)
(:init 
(handempty)
(ontable a)
(ontable b)
(ontable c)
(ontable j)
(ontable f)
(ontable d)
(ontable e)
(ontable k)
(ontable l)
(ontable g)
(ontable i)
(clear a)
(clear b)
(clear c)
(clear j)
(clear f)
(clear d)
(clear e)
(clear k)
(clear l)
(clear g)
(clear i)
)
(:goal
(and
(on a b)
(on b c)
(on c j)
(on j f)
(on f d)
(on d e)
(on e k)
(on k l)
(on l g)
(on g i)
)))