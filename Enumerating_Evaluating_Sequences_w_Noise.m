%-------------------------------------------------------------------------
% define parameters of the noise spectrum in time and frequency domains
%-------------------------------------------------------------------------

omega_max = 10;
omega_c = .001;
domega = 0.01;
N = omega_max/domega; % number of time points
S = zeros(1, N);

dt = pi/omega_max;
t_max = pi/domega;
time = linspace(0, t_max, N);
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------



%-------------------------------------------------------------------------
% Enumerate all possible pulse sequences for Nt equally spaced time points
% for different pulses; either pi_x=1, pi_y=2, or identity (no pulse)=0
%-------------------------------------------------------------------------

Nt=5; % time slots for pulses, divided into t_max/(N+1) free evolution periods

sequence_array = zeros(Nt^2, Nt); 

for ii=0:1:Nt^2-1
    id = dec2base(ii, 2, Nt);
    for jj=1:1:Nt
        sequence_array(ii+1, jj) = str2num(id(jj));
    end
end

avg_coherence = zeros(1, Nt^2);

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

tsteps_FE = int16(N/(Nt+1)); % number of time steps between pulses


Nruns = 20; % number of runs to average over
    
for mm=1:1:Nt^2 % for each pulse sequence

    mm/Nt^2

    for nn=1:1:Nruns % average over runs

        nn;
    
        % generate noise, given noise spectrum (currently lorentzian), with random phase between frequency
        % components
    
        for ii=1:1:N
            S(ii) = (omega_c/(2*pi))/((domega*ii)^2 + (omega_c/2)^2) * exp(2*pi*1i*rand);
        end
            
        noise = zeros(1, N);
        
        for ii=1:1:N
            for jj=1:1:N
                noise(ii) = noise(ii) + (10)*S(jj)*exp(1i*(jj*domega)*(ii*dt))/sqrt(N);
            end
        end
        
        noise = real(noise);
    
        overlap_ramsey = zeros(1, N);
        overlap_SE = zeros(1, N);
        
        % now calculate the evolution under the generated noise
        % Can do either the full SE evolution, or just calculate the phase
        
    %     Id = [1 0; 0 1];
    %     Sz = [1 0; 0 -1];
    %     Sy = [0 -1i; 1i 0];
    %     Sx = [0 1; 1 0];
    %     
    %     psi0 = (1/sqrt(2))*[1; 1];
    %     
    %     psi_ramsey = zeros(2, 1);
    %     psi_ramsey=psi0;
    % 
    %     psi_SE = zeros(2, 1);
    %     psi_SE=psi0;
    %     
    %     
    %     
    %     for ii=1:1:N
    % 
    %         psi_ramsey = psi_ramsey + (-1i*dt)*noise(ii)*Sz*psi_ramsey;
    %         psi_SE = psi_SE + (-1i*dt)*noise(ii)*Sz*psi_SE;
    %         overlap_ramsey(ii)=dot(psi_ramsey, psi0);
    %         overlap_SE(ii)=dot(psi_SE, psi0);
    % 
    %         if ii==N/2
    %             psi_SE = Sx*psi_SE;
    %         end
    % 
    %     end
    
    
    
        phi = 0;
        
        for ii=1:1:N-1 % integrating phase over time period
            phi = phi + noise(ii)*dt;

            % conditional phase manipulation at designated time points

            if mod(ii, tsteps_FE)==0

                kk=ii/tsteps_FE; % this is the pulse we're on at this point

                if sequence_array(mm, kk)==1
                    phi = -phi; % pi_x
                end
%                 if sequence_array(mm, kk)==2
%                     phi = pi-phi; % pi_y
%                 end

            end

        end
    
        avg_coherence(mm) = avg_coherence(mm) + (cos(phi/2)^2)/Nruns;
    
    end


end


[B, I] = sort(avg_coherence);
ordered_sequences = zeros(Nt^2, Nt);

for qq=1:1:Nt^2
    ordered_sequences(qq, :) = sequence_array(I(qq), :);
end

ordered_sequences = transpose(ordered_sequences);
    
t=tiledlayout(2,1)

% Top plot
ax1 = nexttile;
plot(B)
ylabel("Coherence")

% Bottom plot
ax2 = nexttile;
image(ordered_sequences,'CDataMapping','scaled')
c=colorbar;
c.Ticks = [0, 1, 2];
c.TickLabels = {"1", "\pi_x", "\pi_y"};
ylabel("Time point for pulse")


linkaxes([ax1,ax2],'x');


title(t,'Ranking pulse sequences')
xlabel(t,'Rank')

% Move plots closer together
xticklabels(ax1,{})
t.TileSpacing = 'compact';