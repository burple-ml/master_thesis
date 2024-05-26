%Creating peak at 1200
w = linspace(20,2000)
F = rand(1)
Omega = 1200
Q = 70
Type1 = damped_model (w, F, Omega, Q)
for i = 1:20
    F = rand(1)
    a = damped_model(w, F, Omega ,Q)
    Type1 = cat(1, Type1, a)
end
Target1 = ones(1,21) %Target Values

%Creating peak at 800
F = rand(1)
Omega = 800
Type2 = damped_model(w, F, Omega, Q)
for i = 1:20
    F = rand(1)
    a = damped_model(w, F, Omega ,Q)
    Type2 = cat(1, Type2, a)
end
Target2 = 2*ones(1,21)

%Creating a no peak signal
Target3 = zeros(1,21)
Type3 = zeros(21,100)
