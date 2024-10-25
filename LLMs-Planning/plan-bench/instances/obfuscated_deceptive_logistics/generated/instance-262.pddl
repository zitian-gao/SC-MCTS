
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-262)
    (:domain obfuscated_deceptive_logistics)

    (:objects
        o0 o1 o2 o3 o4 o5 o6 o7 o8 - object
    )

    (:init
        (stupendous o0)
        (sneeze o1)
        (cats o2)
        (texture o7)
        (texture o5)
        (texture o3)
        (collect o3 o0)
        (collect o5 o0)
        (collect o7 o0)
        (hand o4)
        (hand o8)
        (hand o6)
        (next o8 o7)
        (next o1 o3)
        (next o6 o5)
        (next o2 o7)
        (next o4 o3)
        (spring o7)
    )

    (:goal
        (and (next o6 o7) (next o4 o5))
    )

    
    
    
)

