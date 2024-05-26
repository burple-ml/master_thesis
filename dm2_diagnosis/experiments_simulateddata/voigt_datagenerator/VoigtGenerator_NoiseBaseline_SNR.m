% voigt parameters
%alpha = 1;  %Peak intensity 
SNR = 10e+4;
sigma = sqrt(1/SNR);
width = 5;  %width of the Peak   
%center = 1200; %Peak frequency   
eta = 0.5;   %the eta factor for the full width half maximum   
%params = [[alpha center width]+randn(1,3) rand(1)];
x = linspace(20,2000,1000); %shift values

baseline_x=1:1000;
baseline_y=1e-03*x;

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
    a = a + sigma*randn(1,size(a,2));
    %add baseline
    a=a+baseline_y;
    Type1 = cat(1, Type1, a);
end
Target1 = ones(1,21); %Target Values

%Creating peak at 800
center = 1200;
width=500;
params = [[alpha center width]+1e-02*randn(1,3) rand(1)];
Type2 = pseudovoigt(params, x);
for i = 1:20
    alpha=rand(1);
    params = [alpha center+5*randn(1) width+randn rand(1)];
    a = pseudovoigt(params, x);
    %add noise
    a = a + sigma*randn(1,size(a,2));
    a = a + baseline_y;
    Type2 = cat(1, Type2, a);
end
Target2 = 2*ones(1,21);

%Creating a no peak signal 
Target3 = zeros(1,21);
Type3 = sigma*randn(21,1000);