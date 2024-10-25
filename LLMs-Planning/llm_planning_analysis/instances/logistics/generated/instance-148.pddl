(define (problem LG-generalization)
(:domain logistics-strips)(:objects c1 t1 a1 l1-4 p7 l1-5 p5 l1-6 p6 l1-0 p4 l1-7 l1-1 l1-8 l1-3 l1-2 c0 t0 a0 l0-4 p3 l0-5 p1 l0-6 p2 l0-0 p0 l0-7 l0-1 l0-8 l0-3 l0-2)
(:init 
(CITY c1)
(TRUCK t1)
(AIRPLANE a1)
(LOCATION l1-4)
(in-city l1-4 c1)
(OBJ p7)
(at p7 l1-4)
(at t1 l1-4)
(LOCATION l1-5)
(in-city l1-5 c1)
(OBJ p5)
(at p5 l1-5)
(LOCATION l1-6)
(in-city l1-6 c1)
(OBJ p6)
(at p6 l1-6)
(LOCATION l1-0)
(in-city l1-0 c1)
(OBJ p4)
(at p4 l1-0)
(LOCATION l1-7)
(in-city l1-7 c1)
(LOCATION l1-1)
(in-city l1-1 c1)
(LOCATION l1-8)
(in-city l1-8 c1)
(LOCATION l1-3)
(in-city l1-3 c1)
(LOCATION l1-2)
(in-city l1-2 c1)
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-4)
(in-city l0-4 c0)
(OBJ p3)
(at p3 l0-4)
(at t0 l0-4)
(LOCATION l0-5)
(in-city l0-5 c0)
(OBJ p1)
(at p1 l0-5)
(LOCATION l0-6)
(in-city l0-6 c0)
(OBJ p2)
(at p2 l0-6)
(LOCATION l0-0)
(in-city l0-0 c0)
(OBJ p0)
(at p0 l0-0)
(LOCATION l0-7)
(in-city l0-7 c0)
(LOCATION l0-1)
(in-city l0-1 c0)
(LOCATION l0-8)
(in-city l0-8 c0)
(LOCATION l0-3)
(in-city l0-3 c0)
(LOCATION l0-2)
(in-city l0-2 c0)
(AIRPORT l1-2)
(at a1 l1-2)
(AIRPORT l0-2)
(at a0 l0-2)
)
(:goal
(and
(at p7 l1-5)
(at p5 l1-6)
(at p6 l1-0)
(at p3 l0-5)
(at p1 l0-6)
(at p2 l0-0)
(at p4 l0-2)
(at p0 l1-2)
)))