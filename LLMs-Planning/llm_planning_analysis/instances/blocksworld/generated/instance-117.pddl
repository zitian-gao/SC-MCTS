(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h d f j i)
(:init 
(handempty)
(ontable h)
(ontable d)
(ontable f)
(ontable j)
(ontable i)
(clear h)
(clear d)
(clear f)
(clear j)
(clear i)
)
(:goal
(and
(on h d)
(on d f)
(on f j)
(on j i)
)))