//+
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 0.01};
//+
Point(2) = {0.2, 0, 0, 0.01};
//+
Point(3) = {0.2, 0.1, 0, 0.01};
//+
Point(4) = {0, 0.1, 0, 0.01};
//+
Point(5) = {0.075, 0.1, 0, 0.01};
//+
Point(6) = {0.125, 0.1, 0, 0.01};
//+
Point(7) = {0.075, 0.00, 0, 0.01};
//+
Point(8) = {0.125, 0.00, 0, 0.01};
//+
Line(1) = {1, 7};
//+
Line(2) = {7, 8};
//+
Line(3) = {8, 2};
//+
Line(4) = {2, 3};
//+
Line(5) = {3, 6};
//+
Line(6) = {6, 5};
//+
Line(7) = {5, 4};
//+
Line(8) = {4, 1};
//+
Line(9) = {5, 7};
//+
Line(10) = {6, 8};
//+
Curve Loop(1) = {7, 8, 1, -9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 9, 2, -10};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {5, 10, 3, 4};
//+
Plane Surface(3) = {3};
//+
Physical Curve("impact", 11) = {6};
//+
Physical Curve("free", 12) = {7, 8, 1, 2, 3, 4, 5};
//+
Physical Surface("block", 13) = {1, 2, 3};
//+
Transfinite Line{1,3,5,7} = 75;
Transfinite Line{2,6} = 50;
Transfinite Line{4,8,9,10} = 100;
Transfinite Surface{1,2,3};
Recombine Surface{1,2,3};

