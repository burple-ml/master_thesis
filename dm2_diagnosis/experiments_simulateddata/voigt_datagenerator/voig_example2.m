% voigt example

alpha = 1;
width = 10;
center = 20;
eta = 0.5;
%params = [[alpha center width]+1e-02*randn(1,3) rand(1)];
params = [[alpha center width]+randn(1,3) rand(1) ];
x = linspace(0, 100);

F = pseudovoigt(params, x);


