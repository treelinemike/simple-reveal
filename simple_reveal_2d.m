% restart
close all; clear all; clc;

% settings
maxHorizonLevel = 1.0;
doSaveFrames = 0;
doProjectUnobservedEstimatesToHorizon = 0;
doIllustrateObs = 1;

% choose an estimation scheme
ESTIMATOR_EKF = 1;
ESTIMATOR_EIF = 2;
ESTIMATOR_PF  = 3;
estimationScheme = ESTIMATOR_PF;

% global flag for plot initialization
global plotCallCount gh_patch gh_truth gh_est gh_rmse gh_truth_dot gh_est_dot revealLevels;
plotCallCount = 0;

% initialize frame count
frameCount = 0;

% data storage
estError = [];
obsCounts = zeros(3,1);
P_hist = [];

% initialize true state (x_t)
xt = [ 0.25 0.35 0.45 0.65 -pi/6 pi/2 ]';
N = length(xt);  % number of states

% state bounds (for random generation)
% formatted for using rmvrnd() from Tim Benham on MATLAB Central:
% https://www.mathworks.com/matlabcentral/fileexchange/34402-truncated-multivariate-normal
% format [ -x1_lower -x2_lower ... -x6_lower x1_upper x2_upper ... x6_upper]'
% NOTE: negative signs on lower bounds! if region is -1 to 1, result is [1 1] (both positive!)
stateBounds = [1 1 0 0 pi pi 1 1 2 2 pi pi]';

% initialize state estimate (x) 
% x0 = [ 0.35 0.25 0.55 0.55 pi/12 pi/3 ]';
% x0 = [ 0.80 0.45 0.35 0.35 -9*pi/8 3*pi/2 ]';
x0 = [ 0.80 0.25 0.5 0.25 3*pi/8 pi/4 ]';
x_prev = x0;

% initialize state estimate error covariance matrix (P)
P0 = diag([0.2 0.2 0.2 0.2 10*pi/180 10*pi/180]);
P_prev = P0;

% initialize process and measurment noise covariance matrices
Q = zeros(length(x0)); %diag([]); % process noise covariance (note: Thrun, et al. calls this R!)
R = diag([0.01 0.01 0.01 0.01 0.01 0.01]); % measurement noise covariance (note: Thrun, et al. calls this Q!)

% initalize information filter variables
% initialize state and measurement covariances
switch(estimationScheme)
    case ESTIMATOR_EKF
        % no parameters to set
        
    case ESTIMATOR_EIF
        M_prev = eye(length(x_prev)); % Omega; Information matrix, initialize to identity
        xi_prev = M_prev*x_prev;      % xi; information vector
        
    case ESTIMATOR_PF
        % set seed for random number generator
        % to get repeatable output (for publication, etc.)
        rng(17);
        
        % number of particles
        M = 5^6;
        
        % generate an initial set of particles using Gaussian
        % approximation/assumption
%         X = mvnrnd(x_prev,P_prev,M)';
        X = rmvnrnd(x_prev,P_prev,M,[-eye(6);eye(6)],stateBounds)';
        
end

% display initial state and iterate through horizon levels
revealLevels = 0:0.01:maxHorizonLevel;
plotRevealModel2d( x_prev, xt, P_prev, revealLevels(1));
for revealLevel = revealLevels
    
    switch(estimationScheme)
        
        case ESTIMATOR_EKF
            
            % compute Jacobian of state transition function
            F = eye(length(x_prev));
            
            % PREDICTION STEP
            % propigate dynamics and error covariance forward in time
            x = x_prev;
            P = F*P_prev*F' + Q;
            
            % compute measurement jacobian
            H = [
                1 0 0         0               0                    0;
                0 1 0         0               0                    0;
                1 0 cos(x(5)) 0              -x(3)*sin(x(5))       0;
                0 1 sin(x(5)) 0               x(3)*cos(x(5))       0;
                1 0 0         cos(x(5)+x(6)) -x(4)*sin(x(5)+x(6)) -x(4)*sin(x(5)+x(6));
                0 1 0         sin(x(5)+x(6))  x(4)*cos(x(5)+x(6))  x(4)*cos(x(5)+x(6))
                ];
            
            % generate an observation
            z_full = simpleRevealObsModel2d( xt );     % full observation is an error-free measurement of all point (x,y) values
            
            % examine each (x,y) pair in the observation
            for pointIdx = 1:(length(z_full)/2)
                
                % coordinates of the current (x,y) pair
                xIdx = 2*pointIdx - 1;
                yIdx = xIdx + 1;
                
                % create a 2-component observation vector, expected observation,
                % and reduce the Jacobian of the measurement model appropriately
                % an alternate approach would be to simply discard unobserved
                % points from z_full and remove the corresponding rows from the
                % H matrix to avoid incremental processing... I believe that this
                % should have the same result
                z_i =     [z_full(xIdx) z_full(yIdx)]';
                z_hat_full = simpleRevealObsModel2d( x );  % expected observation based on current model; this could probably be pre-computed for each timestep (don't recalculate between measurement updates at same timestep) - think the approaches may be equivalent
                z_hat_i = [z_hat_full(xIdx) z_hat_full(yIdx)]';
                H_i = H([xIdx yIdx],:);
                R_i = R([xIdx yIdx],[xIdx yIdx]);
                
                % push model along y axis to horizon if point not observed
                if( (z_hat_i(2) <= revealLevel) && (z_i(2) > revealLevel) && doProjectUnobservedEstimatesToHorizon)
                    disp('push update');
                    delta_z = zeros(size(z_full));
                    delta_z(yIdx) = revealRLevel-z_hat_full(yIdx);
                    x_synth = x + H\delta_z;
                    x = x_synth;
                end
                if((z_i(2) <= revealLevel))
                    disp('normal update');
                    % CORRECTION STEP
                    % update state and error covariance
                    K = P*H_i'/(H_i*P*H_i'+R_i);
                    innov = (z_i - z_hat_i);
                    x = x + K*innov;
                    P = (eye(size(K,1))-K*H_i)*P;
                end
            end
            
            % inflate P to reduce overconfidence
            P = 1.04*P;  
            
            % we will keep x and P around for now
            % they will be put into x_prev and P_prev outside of switch()
            % statement
            
        case ESTIMATOR_EIF
            
            % PREDICT STEP
            % recover previous state estimate
            x_prev = (M_prev)\xi_prev;
            
            % compute process Jacobian and a priori information matrix
            F = eye(length(x_prev));
            M = inv((F/(M_prev))*F'+Q);
            
            % compute a priori state estimate and information vector
            x = x_prev; %f(x) = x ... static system
            xi = M*x;
            
            % compute measurement Jacobian
            H = [
                1 0 0         0               0                    0;
                0 1 0         0               0                    0;
                1 0 cos(x(5)) 0              -x(3)*sin(x(5))       0;
                0 1 sin(x(5)) 0               x(3)*cos(x(5))       0;
                1 0 0         cos(x(5)+x(6)) -x(4)*sin(x(5)+x(6)) -x(4)*sin(x(5)+x(6));
                0 1 0         sin(x(5)+x(6))  x(4)*cos(x(5)+x(6))  x(4)*cos(x(5)+x(6))
                ];
            
            % generate an observation
            z_full = simpleRevealObsModel2d( xt );     % full observation is an error-free measurement of all point (x,y) values
            
            % initialize update accumulation variables
            Msum = zeros(size(M));
            xisum = zeros(size(xi));
            sumIdx = 0;
            
            % examine each (x,y) pair in the observation
            for pointIdx = 1:(length(z_full)/2)
                
                % coordinates of the current (x,y) pair
                xIdx = 2*pointIdx - 1;
                yIdx = xIdx + 1;
                
                % create a 2-component observation vector, expected observation,
                % and reduce the Jacobian of the measurement model appropriately
                % an alternate approach would be to simply discard unobserved
                % points from z_full and remove the corresponding rows from the
                % H matrix to avoid incremental processing... I believe that this
                % should have the same result
                z_i = [z_full(xIdx) z_full(yIdx)]';
                z_hat_full = simpleRevealObsModel2d( x );  % expected observation based on current model; this could probably be pre-computed for each timestep (don't recalculate between measurement updates at same timestep) - think the approaches may be equivalent
                z_hat_i = [z_hat_full(xIdx) z_hat_full(yIdx)]';
                H_i = H([xIdx yIdx],:);
                R_i = R([xIdx yIdx],[xIdx yIdx]);
                
                % perform update if true feature is visible
                if((z_i(2) <= revealLevel))
                    
                    
                    % illustrate observation
                    if(doIllustrateObs)
                        figure(1);
                        subplot(4,1,1:3);
                        hold off;
                        plotRevealModel2d( x, xt, inv(M), revealLevel);
                        hold on; grid on;
                        plot(z_i(1),z_i(2),'go','MarkerSize',20,'LineWidth',2);
                        plot(z_hat_i(1),z_hat_i(2),'g*','MarkerSize',20,'LineWidth',2);
                        pause(1)
                    end
                    
                    % add to measurement update information matrix and vector
                    Msum = Msum + (H_i'/(R_i))*H_i;
                    xisum = xisum + (H_i'/(R_i))*(z_i-z_hat_i+H_i*x);
                    sumIdx = sumIdx + 1;
                end
            end
            
            % CORRECTION STEP
            % apply measurement update if we actually made measurements
            if(sumIdx ~= 0)
                disp('applying updates');
                M = M + Msum;
                xi = xi + xisum;
            end
            
            P = inv(M);
            x = P*xi;
            M_prev = M;
            xi_prev = xi;
            
            % we will keep x and P around for now
            % they will be put into x_prev and P_prev outside of switch()
            % statement
            
        case ESTIMATOR_PF
            
            % (re)initialize all weights and a priori particles to zero
            q = zeros(M,1);
            X_prior = zeros(N,M);
            
            % plot current particle set (2D, just location of elbow particle)
            figure(3);
            set(gcf,'Position',[4.634000e+02 3.418000e+02 1052 4.200000e+02]);
            subplot(1,3,1);
            hold off;
            plot(X(1,:),X(2,:),'b.','MarkerSize',20);
            xlim(getWiderBounds(-stateBounds(1),stateBounds(1+6)));
            ylim(getWiderBounds(-stateBounds(2),stateBounds(2+6)));
            hold on; grid on;
            plot(xt(1),xt(2),'ro','MarkerSize',10,'LineWidth',2);
            plot(x0(1),x0(2),'k.','MarkerSize',20,'LineWidth',2);
            plot(x_prev(1),x_prev(2),'m+','MarkerSize',10,'LineWidth',2);
            xlabel('x_0');
            ylabel('y_0');
            axis square;

            subplot(1,3,2);
            hold off;
            plot(X(3,:),X(4,:),'b.','MarkerSize',20);
            xlim(getWiderBounds(-stateBounds(3),stateBounds(3+6)));
            ylim(getWiderBounds(-stateBounds(4),stateBounds(4+6)));
            hold on; grid on;
            plot(xt(3),xt(4),'ro','MarkerSize',10,'LineWidth',2);
            plot(x0(3),x0(4),'k.','MarkerSize',20,'LineWidth',2);
            plot(x_prev(3),x_prev(4),'m+','MarkerSize',10,'LineWidth',2);
            xlabel('r_1');
            ylabel('r_2');
            axis square;
            
            subplot(1,3,3);
            hold off;
            plot(X(5,:),X(6,:),'b.','MarkerSize',20);
            xlim(getWiderBounds(-stateBounds(5),stateBounds(5+6)));
            ylim(getWiderBounds(-stateBounds(6),stateBounds(6+6)));
            hold on; grid on;
            plot(xt(5),xt(6),'ro','MarkerSize',10,'LineWidth',2);
            plot(x0(5),x0(6),'k.','MarkerSize',20,'LineWidth',2);
            plot(x_prev(5),x_prev(6),'m+','MarkerSize',10,'LineWidth',2);
            xlabel('\Theta_1');
            ylabel('\Theta_2');     
            axis square;
            
            % get a measurement, subject to horizon/occlusion
            z = simpleRevealObsModel2d( xt );
            obs_mask = repelem(z(2:2:end,:) < revealLevel,2);
            Rt = R(obs_mask,obs_mask);
            
            % iterate filter only when we are processing a measurement (the
            % time update is essentially the identity, i.e. forward dyanmics do nothing)
            if(~isempty(Rt))
                
                % loop through particles computing weights based on how well data fits
                % model
                for m = 1:M
                    
                    % get the appropriate particle
                    x_prev = X(:,m);
                    
                    % compute proposal (prediction) for this particle
                    x_prior = x_prev;  % identity transformation
                    X_prior(:,m) = x_prior;
                    
                    % remove unobserved rows from innovation and R matrix
                    z_hat = simpleRevealObsModel2d( x_prior );
                    innov = z-z_hat;
                    innov = innov(obs_mask);
                                        
                    % compute importance factor / weight
                    % don't need constant coefficient b/c normalizing to sum to one
                    % later...
                    q(m) = exp( -0.5 * innov' * inv(Rt) * innov);  % the q vector incorporates MEASUREMENT data
                    
                end
                
                % normalize weights
                q = q/sum(q);
                qCDF = cumsum(q);
                
                % starting point: Simon pg. 473: regularized particle filtering
                % first sample particle based on their weights
                x_post_idx = arrayfun(@(afin1) find(afin1 < qCDF,1,'first'), rand(M,1)-eps );  % eps subtracted from rand(M,1) to exclude exactly 1.0000... though this is extremely unlikely
                X_post_prereg = X_prior(:,x_post_idx);
                
                % compute particle ensemble mean and covariance matrix
                % mean
                mu_prereg = mean(X_post_prereg,2); % equivalent to: (1/M)*sum(X_prior,2);
                
                % covariance
                S_prereg = zeros(N);
                for mIdx = 1:M
                    S_prereg = S_prereg + (X_post_prereg(:,mIdx)-mu_prereg)*(X_post_prereg(:,mIdx)-mu_prereg)';
                end
                S_prereg = (1/(M-1))*S_prereg;
                
                % generate new posterior particles using Gaussian approximation to
                % posterior
%                 X_post = mvnrnd(mu_prereg,1.1*S_prereg,M)';
                X_post = rmvnrnd(mu_prereg,1.04*S_prereg,M,[-eye(6);eye(6)],stateBounds)';
                
                % update with the posteriror particle set
                X = X_post;   % THIS PREPARES PARTICLE SET FOR NEXT ITERATION
            end
            
            % mean of posterior particle set
            mu_post = mean(X,2);
            
            % covariance of posterior particle set
            P_post = zeros(N);
            for mIdx = 1:M
                P_post = P_post + (X(:,mIdx)-mu_post)*(X(:,mIdx)-mu_post)';
            end
            P_post = (1/(M-1))*P_post;
            
            % update x and P to conform with other estimation schemes
            x = mu_post;
            P = P_post;
            
        otherwise
            disp('Invalid assimilation algorithm choice.');
    end
    
    % store P
    P_hist(end+1,:) = reshape(P',[],1)';
    
    % display current estimate and true state
    plotRevealModel2d( x, xt, P, revealLevel);
    
    % save frames for animation
    % then convert - not in MATLAB although could use system() command - using a codec compatible with LaTeX (Beamer)
    % see: https://tex.stackexchange.com/questions/141622/trouble-using-ffmpeg-to-encode-a-h264-video-in-a-mp4-container-to-use-with-medi
    % first download FFMPEG for windows: https://ffmpeg.zeranoe.com/builds/ OR https://www.ffmpeg.org/download.html
    % ffmpeg -r 10 -start_number 1 -i frame%003d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4
    if(doSaveFrames)
        saveas(gcf,sprintf('frame%03d.png',frameCount));
        frameCount = frameCount + 1;
    end
    
    % show covariance or information matrix
    
    switch estimationScheme
        case ESTIMATOR_EKF
            figure(2);
            image(P);
        case ESTIMATOR_EIF
            figure(2);
            image(M);
    end
    
    % increment timestep
    x_prev = x;
    P_prev = P;
    
end


% display error covariance trends
figure;
set(gcf,'Position',[0069 0065 1.262400e+03 6.696000e+02]);
for plotIdx = 1:size(P_hist,2)
    rowNum = ceil(plotIdx/N);
    colNum = mod(plotIdx-1,N)+1;
    if( colNum >= rowNum)
        subplot(size(P,1),size(P,2),plotIdx);
        plot(P_hist(:,plotIdx),'b-','LineWidth',1.6);
        hold on; grid on;
    end
end


% convert model parameters into (x,y) locations of each point
function z = simpleRevealObsModel2d( x )

x0 = x(1);
y0 = x(2);
x1 = x0 + x(3)*cos(x(5));
y1 = y0 + x(3)*sin(x(5));
x2 = x0 + x(4)*cos(x(5)+x(6));
y2 = y0 + x(4)*sin(x(5)+x(6));

z = [x0,y0,x1,y1,x2,y2]';

end

% display estimated and true models along with current level of occlusion
function plotRevealModel2d( estParams, trueParams, P, horizonLevel)

global plotCallCount gh_patch gh_truth gh_est gh_rmse gh_truth_dot gh_est_dot revealLevels;

% compute observations and RMSE
z_truth = simpleRevealObsModel2d(trueParams);
z_est = simpleRevealObsModel2d(estParams);
rmse = computeReveal2dError(estParams, trueParams);

% activate animation figure
figure(1);

if(~plotCallCount)
    
    % configure plot
    set(gcf,'Position',[1.034000e+02 2.274000e+02 0356 0540]);
    subplot(4,1,1:3);
    hold on; grid on;
    % plot lines and occlusion patch
    gh_truth = line(z_truth([5 1 3]),z_truth([6 2 4]),'Marker','.','MarkerSize',75,'LineStyle','-','Color',[0 0 1],'LineWidth',2.0);
    gh_truth_dot = line(z_truth(5),z_truth(6),'Marker','.','MarkerSize',25,'Color',[1 1 1]);
    gh_est = line(z_est([5 1 3]),z_est([6 2 4]),'Marker','.','MarkerSize',60,'LineStyle',':','Color',[1 0 0],'LineWidth',2.0);
    gh_est_dot = line(z_est(5),z_est(6),'Marker','.','MarkerSize',25,'Color',[1 1 1]);
    gh_patch = patch([0 1 1 0], [horizonLevel horizonLevel 1.4 1.4],'k','FaceAlpha',0.2,'EdgeColor','none');
    
    % more plot configuraiton
    legh = legend([gh_truth gh_est gh_patch],{'Truth','Estimate','Occlusion'},'Location','northoutside','Orientation','horizontal','FontWeight','bold');
    legh.Position = legh.Position + [.015 -0.015 0 0.03];
    ylim([0 1.2]);
    % axis equal;
    xlim([0 1]);
    axh = gca;
    axh.XAxis.Visible = 'off';
    axh.YAxis.Visible = 'off';
    axh1 = gca;
    axh1.GridColor = 0.4*ones(1,3);
    axh1.GridAlpha = 0.7;
    
    % display RMSE
    rmsePlotVals = [revealLevels' nan(length(revealLevels),1)];
    rmsePlotVals(1,2) = rmse;
    subplot(4,1,4);
    gh_rmse = plot(rmsePlotVals(:,1),rmsePlotVals(:,2),'b-','LineWidth',1.6);
    xlabel('\bfHorizon Level');
    ylabel('\bfRMSE');
    xlim([0 max(revealLevels)]);
    ylim([0 0.65]);
    grid on;
else
    % efficiently update animation plot on subsequent calls to plotRevealModel2d
    gh_truth.XData = z_truth([5 1 3]);
    gh_truth.YData = z_truth([6 2 4]);
    gh_truth_dot.XData = z_truth(5);
    gh_truth_dot.YData = z_truth(6);
    gh_est.XData = z_est([5 1 3]);
    gh_est.YData = z_est([6 2 4]);
    gh_est_dot.XData = z_est(5);
    gh_est_dot.YData = z_est(6);
    gh_patch.Vertices(1,2) = horizonLevel;
    gh_patch.Vertices(2,2) = horizonLevel;
    gh_rmse.YData(plotCallCount+1) = rmse;
end


% % plot uncertainty ellipse for point 0
% xycov = P(1:2,1:2);
% [V,D] = eig(xycov);
% d = diag(D);
% for dirIdx = 1:2
%     plot(estParams(1)+[0,d(dirIdx)*V(1,dirIdx)],estParams(2)+[0,d(dirIdx)*V(2,dirIdx)],'m-');
% end

% ensure that figure updates
drawnow;

% increment plot call counter (needed for skipping initialization and updating RMSE plot)
plotCallCount = plotCallCount + 1;

end

% compute root mean square error between a specific model configuration (x)
% and the true configuration (xt)
function rmse = computeReveal2dError(x,xt)

z_hat = simpleRevealObsModel2d(x);
z_t   = simpleRevealObsModel2d(xt);
z_d = z_t-z_hat;
rmse = sqrt( (z_d'*z_d)/(length(z_d)/2));

end

function widerInterval = getWiderBounds(a,b)
    span = b-a;
    widerInterval = [a-0.1*span b+0.1*span];
end