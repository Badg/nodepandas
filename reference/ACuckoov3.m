function [p,F,pg,evalHist,diversity] = ACuckoov3(K, NestI, S, vardef)
    %MCS Modified Cuckoo Search


    %Written by Sean Walton (512465@swansea.ac.uk) 2011 for Swansea university

    %Please cite the following paper when using this code...
    %S.Walton, O.Hassan, K.Morgan and M.R.Brown "Modified cuckoo search: A
    %new gradient free optimisation algorithm" Chaos, Solitons & Fractals Vol
    %44 Issue 9, Sept 2011 pp. 710-718 DOI:10.1016/j.chaos.2011.06.004
    % 
    % Copyright 2011 Sean Walton sean.walton84@gmail.com
    % This program is distributed under the terms of the GNU General Public License

    % You cannot integrate MCS in any closed-source software you plan to
    % distribute in anyway for any reason.  If you want to integerate MCS into
    % a closed-source software, or want to sell a modified closed source
    % version of MCS contact Sean Walton directly to discuss obtaining a
    % different license
    % 
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU General Public License as published by
    % the Free Software Foundation, version 2.
    % 
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU General Public License for more details.
    % 
    % You should have received a copy of the GNU General Public License
    % along with this program.  If not, see <http://www.gnu.org/licenses/>.


    % ----  Inputs  ----
    %   generations = total number of generations; NestI = initial positions of nests
    % vardef gives the upper and lower bounds of the particle position
    % size(vardef) = (2,No dimensions)  vardef(1,:) = upper bounds, vardef(2,:)
    % = lower bound.
    % S is a structure containing parameters, the following are suggested
    % values

    % S.pa = 0.7;         %Fraction discard
    % S.plot = 0;         %If you want the results plotted set this to 1
    % S.constrain = 1;    %Set to 1 if you want the search constrained within vardef, zero otherwise
    % S.A = 100; %        %Step size factor, increase this to decrease step
    % size
    % S.pwr = 0.5;   %Power that step size is reduced by each generation
    % S.flight = 1;   %Type of random walk
    % S.NesD = 1;    %Number of eggs deleated each generation


    % ----- Outputs -----

    % p = time history of nest position
    % F = time history of objective function value of each nest
    % pg = optimum position found
    % evalHist = Number of funtion evaluations each generation
    % diversity = a diversity measures

    %Counter to count number of objective function evals
    obj_evals = 0;

    f = S.fname;
    A = (vardef(1,:)-vardef(2,:))./S.A;

    timeStart = tic;

    %Find number of dimensions and number of nests
    [NoNests,NoDim] = size(NestI);
    MinNests = 10;

    % Allocate cells to hold the time history of nest position and objective
    % function value
    p = cell(generations,1);
    F = cell(generations,1);


    diversity = zeros(generations,1);						
    evalHist = zeros(generations,1);

    % Allocate matrices for current nest position and fitness
    pi = zeros(NoNests,NoDim);
    Fi = zeros(NoNests,1);
    ptemp = zeros(1,NoDim);
    pmean = zeros(1,NoDim);

    %1 - Calculate fitness for initial nests
    for i = 1:NoNests
        pi(i,:) = NestI(i,:);
        Ftemp = feval(f,pi(i,:));
        obj_evals = obj_evals+1;
        if isreal(Ftemp)
            Fi(i,1) = Ftemp;
        else
            Fi(i,1) = realmax;
        end
        
    end

    %Record values in cells
    p{1,1} = pi;
    F{1,1} = Fi;

    %Calculate diversity

    Ldiag = 0;							

    for ii=1:NoDim										%
        pmean(ii) = mean(pi(:,ii));
    end
    for ii=1:NoNests
        
        distT = norm(pi(ii,:)-pmean);
        Ldiag = max(Ldiag,distT);
        
        diversity(1,1) = diversity(1,1) + distT;
        
    end

    diversity(1,1) = diversity(1,1)/(Ldiag*NoNests);	
    evalHist(1,1) = obj_evals;


    %Plotting statement
    %--------------------
    if eq(S.plot,1)
        %Plot positions
        FPlot(1,1) = min(Fi);
        figure(1)
        clf
        subplot(2,1,1)
        plot(pi(:,1),pi(:,2),'o')
        set(gca,'XLim',[vardef(2,1) vardef(1,1)],'YLim',[vardef(2,2) vardef(1,2)])
        subplot(2,1,2)
        plot(FPlot(1,1),'-+r')
        %set(gca,'YScale','log')
        refreshdata
        drawnow
    end

    pa = S.pa;               
    ptop = 1-pa;                           


    G = 1;
    %Itterate over all generations
    while G<generations
        %This save statement is useful for when objective functions are
        %expensive and the optimiser needs to be run a long time
        
        %save MCSout.mat
    	
        timeElapsed = toc(timeStart);
        G = G+1;
        
        

        %a) sort the current nests in order of fitness
        
        %First put vectors into matrix form
        piFi = [pi Fi];
        %Sort by Fi in assending order
        piFiS = sortrows(piFi,NoDim+1);
        
        %Decrease number of nests, we only need lots of nests initially to get
        %a good sampling of the objective function
        NoNests = max(MinNests,NoNests-S.NesD);
        pi = piFiS(1:NoNests,1:NoDim);
        Fi = piFiS(1:NoNests,NoDim+1);
        NoTop = max(3,round(NoNests*ptop));     %%%%%%%%%%%%%%%
        NoDiscard = NoNests-NoTop;              %%%%%%%%%%%%%%
        
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CURRENT READ MARKER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %2 - Loop over each Cuckoo which has been discarded
        for i = 1:NoDiscard
            a = A./(G^(S.pwr));
            
            %a) Random walk
            NoSteps = 1;
            
            dx = levi_walk(NoSteps,NoDim,S);
            
            for j = 1:NoDim
                %Random number to determine direction
                if rand(1)>0.5
                    ptemp(1,j) = a(1,j)*dx(1,j) + pi(NoNests - i + 1, j);
                else
                    ptemp(1,j) = a(1,j)*dx(1,j) - pi(NoNests - i + 1, j);
                end
            end
           
            %Check position is inside bounds
            upper = gt(ptemp,vardef(1,:));
            lower = lt(ptemp,vardef(2,:));
            if and(eq(S.constrain,1),or(any(upper),any(lower)))
                %Then point is outside bound so don't continue
            else
                %Point valid so update fitness
                %b) Calculate fitness of egg at ptemp
                
                Ftemp = feval(f,ptemp);
                obj_evals = obj_evals+1;
                if isreal(Ftemp)
                    pi(NoNests - i + 1, :) = ptemp;
                    Fi(NoNests - i + 1, 1) = Ftemp;
                else
                end
                
            end
        end
        
      
        
        
        %3 - Loop over each Cuckoo not to be Discarded
        for C = 1:(NoTop)
            
            %Pick one of the top eggs to cross with
            randNest = round(1 + (NoTop-1).*rand(1));
            
            if randNest == C
                % Cross with egg outside elite
                randNest = NoNests-round(1 + (NoDiscard-1).*rand(1));
                
                dist(1,:) = pi(randNest,:) - pi(C,:);
                %Multiply distance by a random number
                
                dist = dist.*rand(1,NoDim);
                ptemp = pi(C,:) + dist(1,:);
                
                if ismember(ptemp,pi,'rows')
                    
                    % Perform random walk instead
                    
                    a = A./(G^(2*S.pwr));
                    NoSteps = 1;
                    
                    dx = levi_walk(NoSteps,NoDim,S);
                    
                    for j = 1:NoDim
                        %Random number to determine direction
                        if rand(1)>0.5
                            ptemp(1,j) = a(1,j)*dx(1,j) + pi(randNest, j);
                        else
                            ptemp(1,j) = a(1,j)*dx(1,j) - pi(randNest, j);
                        end
                    end
                end
                
            else
                if Fi(randNest,1)>Fi(C,1)
                    
                    %Calculate distance
                    dist(1,:) = pi(C,:) - pi(randNest,:);
                    %Multiply distance by a random number
                    
                    dist = dist.*rand(1,NoDim);
                    ptemp = pi(randNest,:) + dist(1,:);
                    
                    if ismember(ptemp,pi,'rows')
                        
                        % Cross with egg outside elite
                        CI = NoNests-round(1 + (NoDiscard-1).*rand(1));
                        
                        dist(1,:) = pi(randNest,:) - pi(CI,:);
                        %Multiply distance by a random number
                        
                        dist = dist.*rand(1,NoDim);
                        ptemp = pi(CI,:) + dist(1,:);
                        
                        if ismember(ptemp,pi,'rows')
                            
                            % Perform random walk instead
                            
                            a = A./(G^(2*S.pwr));
                            NoSteps = 1;
                            
                            dx = levi_walk(NoSteps,NoDim,S);
                            
                            for j = 1:NoDim
                                %Random number to determine direction
                                if rand(1)>0.5
                                    ptemp(1,j) = a(1,j)*dx(1,j) + pi(randNest, j);
                                else
                                    ptemp(1,j) = a(1,j)*dx(1,j) - pi(randNest, j);
                                end
                            end
                            
                            
                        end
                        
                    end
                    
                    
                elseif Fi(C,1)>Fi(randNest,1)
                    %Search in direction of randNest by golden ratio
                    %Calculate distance
                    dist(1,:) = pi(randNest,:) - pi(C,:);
                    %Multiply distance by a random number
                    
                    dist = dist.*rand(1,NoDim);
                    ptemp = pi(C,:) + dist(1,:);
                    
                    if ismember(ptemp,pi,'rows')
                        
                        % Cross with egg outside elite
                        randNest = NoNests-round(1 + (NoDiscard-1).*rand(1));
                        
                        dist(1,:) = pi(randNest,:) - pi(C,:);
                        %Multiply distance by a random number
                        
                        dist = dist.*rand(1,NoDim);
                        ptemp = pi(C,:) + dist(1,:);
                        
                        if ismember(ptemp,pi,'rows')
                            
                            % Perform random walk instead
                            
                            a = A./(G^(2*S.pwr));
                            NoSteps = 1;
                            
                            dx = levi_walk(NoSteps,NoDim,S);
                            
                            for j = 1:NoDim
                                %Random number to determine direction
                                if rand(1)>0.5
                                    ptemp(1,j) = a(1,j)*dx(1,j) + pi(randNest, j);
                                else
                                    ptemp(1,j) = a(1,j)*dx(1,j) - pi(randNest, j);
                                end
                            end
                            
                            
                        end
                        
                    end
                    
                else
                   
                    
                    dist(1,:) = pi(randNest,:) - pi(C,:);
                    %Multiply distance by a random number
                    dist = dist.*rand(1,NoDim);
                    ptemp = pi(C,:) + dist(1,:);
                    
                    if ismember(ptemp,pi,'rows')
                        
                        % Cross with egg outside elite
                        randNest = NoNests-round(1 + (NoDiscard-1).*rand(1));
                        
                        dist(1,:) = pi(randNest,:) - pi(C,:);
                        %Multiply distance by a random number
                        
                        dist = dist.*rand(1,NoDim);
                        ptemp = pi(C,:) + dist(1,:);
                        
                        if ismember(ptemp,pi,'rows')
                            
                            % Perform random walk instead
                            
                            a = A./(G^(2*S.pwr));
                            NoSteps = 1;
                            
                            dx = levi_walk(NoSteps,NoDim,S);
                            
                            for j = 1:NoDim
                                %Random number to determine direction
                                if rand(1)>0.5
                                    ptemp(1,j) = a(1,j)*dx(1,j) + pi(randNest, j);
                                else
                                    ptemp(1,j) = a(1,j)*dx(1,j) - pi(randNest, j);
                                end
                            end
                            
                            
                        end
                        
                    end
                end
            end
            
            %Check position is inside bounds
            upper = gt(ptemp,vardef(1,:));
            lower = lt(ptemp,vardef(2,:));
            if and(eq(S.constrain,1),or(any(upper),any(lower)))
                %Then point is outside bound so don't continue
            else
                %Point valid so update fitness
                %b) Calculate fitness of egg at ptemp
                
                Ftemp = feval(f,ptemp);
                obj_evals = obj_evals+1;
                
                %c) Select random nest and replace/update position if fitness is
                %better
                
                %Select random index
                randNest = round(1 + (NoNests-1).*rand(1));
                
                if and((Fi(randNest,1)>Ftemp),isreal(Ftemp));
                    %Then replace egg
                    pi(randNest,:) = ptemp;
                    Fi(randNest,1) = Ftemp;
                    
                else
                    %Discard new egg
                end
            end
            
            
        end
        

        
         %2a) Emptying routine from yang and deb
        new_nest=empty_nests(pi,pa);
        for i=1:size(pi,1)
            ptemp = new_nest(i,:);
             %Check position is inside bounds
            upper = gt(ptemp,vardef(1,:));
            lower = lt(ptemp,vardef(2,:));
            
            
            
            
            if and(eq(S.constrain,1),or(any(upper),any(lower)))
                %Then point is outside bound so don't continue
            else
                
                if not(ismember(ptemp,pi,'rows'))
                    
                    %Point valid so update fitness
                    %b) Calculate fitness of egg at ptemp
                    fold = Fi(i,1);
                    Ftemp = feval(f,ptemp);
                    obj_evals = obj_evals+1;
                    if and(isreal(Ftemp),lt(Ftemp,fold))
                        pi(i, :) = ptemp;
                        Fi(i, 1) = Ftemp;
                    else
                    end
                end
            end
        end
        
        
        %Record values in cells
        p{G,1} = pi;
        F{G,1} = Fi;
        
        %Calculate diversity (distance-to-average-point)                 %%%%%%%%%%%%%%%%
        
        for ii=1:NoDim
            pmean(ii) = mean(pi(:,ii));
        end
        Ldiag = 0;
        for ii=1:NoNests
            distT = norm(pi(ii,:)-pmean);
            Ldiag = max(Ldiag,distT);
            diversity(G,1) = diversity(G,1) + norm(pi(ii,:)-pmean);    %%%%%%%%%%%%%%%%%%%%
            
        end
        
        diversity(G,1) = diversity(G,1)/(Ldiag*NoNests);
        evalHist(G,1) = obj_evals;
        if eq(S.plot,1)
            %Plot positions
            
            FPlot(G,1) = min(Fi);
            figure(1)
            clf
            hold on
            subplot(2,1,1)
            plot(pi(:,1),pi(:,2),'o')
            set(gca,'XLim',[vardef(2,1) vardef(1,1)],'YLim',[vardef(2,2) vardef(1,2)])
            subplot(2,1,2)
            plot(FPlot(1:G,1),'-+r')
            %set(gca,'YScale','log')
            refreshdata
            drawnow
        end
        
        
        
        
    end


    %Find best solution
    [Fg,ind] = min(Fi);

    pg = pi(ind,:);


    end

function [dx] = levi_walk(NoSteps,NoDim,S)
    %Function to produce a levi random walk of NoSteps steps in NoDim dimensions

    %Allocate matrix for solutions

    dx = zeros(1,NoDim);


    if eq(S.flight,1)
        
        
        %Yang-Deb levy
        
        beta=3/2;
        sigma=
           (gamma(1+beta)*sin(pi*beta/2)/
            (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
        u = randn(1,NoDim)*sigma;
        v = randn(1,NoDim);
        dx = u./abs(v).^(1/beta);
        
        
    elseif eq(S.flight,2)
        %Cauchy Levy FLight
        for i=1:NoDim
            x = gen_levy_flight(NoSteps,1,1,0,0,'isotropic');
            dx(1,i) = x(1,1);
        end
        
    elseif eq(S.flight,3)
        %Gaussian walk
         for i=1:NoDim
            x = gen_levy_flight(NoSteps,2,1,0,0,'isotropic');
            dx(1,i) = x(1,1);
         end
    else
        %Uniformally distributed
        dx = rand(1,NoDim);
    end

    end

function x = gen_levy_flight(n,alpha,sigma,beta,delta,type)

    % gen_levy_flight - generate a Levy flight
    %
    %   x = gen_levy_flight(n,alpha,sigma,beta,delta,type);
    %
    %   n is the length
    %   alpha is the exponent (alpha=2 for gaussian, alpha=1 for cauchian)
    %   sigma is the standard deviation
    %   beta and delta are symmetry parameter (for no drift, set to 0)
    %   type is either 'isotropic' or 'axis'
    %
    %   Copyright (c) 2005 Gabriel Peyr


    if nargin<2
        alpha = 1;
    end
    if nargin<3
        sigma = 1;
    end
    if nargin<4
        beta = 0;
    end
    if nargin<5
        delta = 0;
    end
    if nargin<6
        type = 'isotropic';
    end

    x = zeros(n,2);
    if strcmp(type, 'isotropic')
        r = stabrnd(alpha, beta, sigma, delta, 1, n);
        theta = 2*pi*rand(1,n);
        x(:,1) = r.*cos(theta);
        x(:,2) = r.*sin(theta);
    else
        x(:,1) = stabrnd(alpha, beta, c, delta, 1, n);
        x(:,2) = stabrnd(alpha, beta, c, delta, 1, n);
    end
    x = cumsum(x);
    end

function new_nest = empty_nests(nest,pa)
    % A fraction of worse nests are discovered with a probability pa
    n=size(nest,1);
    % Discovered or not -- a status vector
    K=rand(size(nest))>pa;

  
    %% New solution by biased/selective random walks
    stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
    new_nest=nest+stepsize.*K;
    end

