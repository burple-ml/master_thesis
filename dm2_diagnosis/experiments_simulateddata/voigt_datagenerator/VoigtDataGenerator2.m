% voigt parameters
%alpha = 1;  %Peak intensity 
width = 5;  %width of the Peak   
%center = 1200; %Peak frequency   
eta = 0.5;   %the eta factor for the full width half maximum   
%params = [[alpha center width]+randn(1,3) rand(1)];
x = linspace(20,2000,1000); %shift values

%Creating Signal with peak at 1200
center = 1200;
alpha = rand(1);
params = [[alpha center width]+1e-02*randn(1,3) rand(1)];
Type1 = pseudovoigt(params, x);
for i = 1:20
    alpha=rand(1);
    params = [alpha center+5*randn(1) width+randn rand(1)];
    a = pseudovoigt(params, x);
    %add noise
    a = a + 1e-2*randn(1,size(a,2));
    Type1 = cat(1, Type1, a);
end
Target1 = ones(1,21); %Target Values

%Creating peak at 800
center = 800;
params = [[alpha center width]+1e-02*randn(1,3) rand(1)];
Type2 = pseudovoigt(params, x);
for i = 1:20
    alpha=rand(1);
    params = [alpha center+5*randn(1) width+randn rand(1)];
    a = pseudovoigt(params, x);
    %add noise
    a = a + 1e-2*randn(1,size(a,2));
    Type2 = cat(1, Type2, a);
end
Target2 = 2*ones(1,21);

%Creating a no peak signal 
Target3 = zeros(1,21);
Type3 = zeros(21,1000);