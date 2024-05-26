function [B] = damped_model(w, F, Omega, Q)
    % DAMPED_MODEL
    % w -- Frequencies
    % F -- Amplitude at peak location
    % Omega -- Peak location
    % Q -- Q-factor
    
    B = F ./sqrt((Omega.^2 - w.^2).^2 + (Omega.^2).*(w.^2)./(Q.^2));
    B = (Omega.^2 ./ Q) .* B;
    
end

