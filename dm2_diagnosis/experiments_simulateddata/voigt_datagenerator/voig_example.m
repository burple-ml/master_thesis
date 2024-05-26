% voigt example

alpha = 10;
width = 10;
center = 20;
eta = 0.5;
params = [[alpha center width]+randn(1,3) rand(1)];
x = linspace(0, 100);

F = pseudovoigt(params, x);
