% particle filter test

% restart
close all; clear all; clc;

% set seed for random number generator
% to get repeatable output (for publication, etc.)
rng(17);

% global flag for plot initialization
global plotCallCount gh_patch gh_truth gh_est gh_rmse gh_truth_dot gh_est_dot revealLevels;
plotCallCount = 0;
revealLevels = 0:0.01:1;
maxHorizonLevel = 1.0;

M = 5^6; % number of particles

% initialize true state (x_t)
xt = [ 0.25 0.35 0.45 0.65 -pi/6 pi/2 ]';

N = length(xt);  % number of states

% initialize state estimate (x)
% x0 = [ 0.35 0.25 0.55 0.55 pi/12 pi/3 ]';
% x0 = [ 0.80 0.45 0.35 0.35 -9*pi/8 3*pi/2 ]';
x0 = [ 0.80 0.25 0.5 0.25 3*pi/8 pi/4 ]';
x = x0;

% initialize state error covariance matrix (P)
P0 = diag([0.2 0.2 0.2 0.2 2*pi/180 2*pi/180]);
P = P0;

% set measurement noise covariance (note: Thrun, et al. calls this Q!)
R = diag([0.05 0.05 0.05 0.05 0.05 0.05]);

% generate an initial set of particles using Gaussian
% approximation/assumption
X = mvnrnd(x,P,M)';

% display initial state


% run repeated observations & display initial state
revealLevels = 0:0.01:maxHorizonLevel;
plotRevealModel2d( x, xt, P, revealLevels(1));
for revealLevel = revealLevels
    for repeatObsIdx = 1:1
        % initialize all weights and a priori particles to zero
        q = zeros(M,1);
        X_prior = zeros(N,M);
        
        
        % plot current particle set (2D, just location of elbow particle)
        figure(2);
        hold off;
        plot(X(1,:),X(2,:),'b.','MarkerSize',20);
        xlim([0 2]);
        ylim([0 2]);
        hold on;
        plot(xt(1),xt(2),'ro','MarkerSize',10,'LineWidth',2);
        plot(x0(1),x0(2),'k.','MarkerSize',20,'LineWidth',2);
        
        
        % get a measurement
        z = simpleRevealObsModel2d( xt );
        obs_mask = repelem(z(2:2:end,:) < revealLevel,2);
        Rt = R(obs_mask,obs_mask);
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
                
                
                % compute importance factor
                % don't need constant coefficient b/c normalizing to sum to one
                % later...
                q(m) = exp( -0.5 * innov' * inv(Rt) * innov);  % the q vector incorporates MEASUREMENT data
                
            end
            
            % normalize weights
            q = q/sum(q);
            qCDF = cumsum(q);
            
            
            
            % starting point: Simon pg. 473: regularized particle filtering
            
            
            % sample particle based on their weights
            x_post_idx = arrayfun(@(afin1) find(afin1 < qCDF,1,'first'), rand(M,1)-eps );  % eps subtracted from rand(M,1) to exclude exactly 1.0000... though this is extremely unlikely
            X_post_prereg = X_prior(:,x_post_idx);
            
            
            % compute particle ensemble mean and covariance matrix
            
            % mean
            mu = mean(X_post_prereg,2); % equivalent to: (1/M)*sum(X_prior,2);
            
            % covariance
            S = zeros(N);
            for mIdx = 1:M
                S = S + (X_post_prereg(:,mIdx)-mu)*(X_post_prereg(:,mIdx)-mu)';
            end
            S = (1/(M-1))*S;
            
            % generate new posterior particles using Gaussian approximation to
            % posterior
            X_post = mvnrnd(mu,1.1*S,M)';           
            
            % update particle set
            X = X_post;
        end
        
        x_post = mean(X,2);
        
        % show new mean
        figure(2)
        plot(x_post(1),x_post(2),'m*','MarkerSize',20);
        
        % display result
        plotRevealModel2d( x_post, xt, P, revealLevel);
        drawnow;
        
    end
end

% convert model parameters into (x,y) locations of each point
function z_hat = simpleRevealObsModel2d( x )

x0 = x(1);
y0 = x(2);
x1 = x0 + x(3)*cos(x(5));
y1 = y0 + x(3)*sin(x(5));
x2 = x0 + x(4)*cos(x(5)+x(6));
y2 = y0 + x(4)*sin(x(5)+x(6));

z_hat = [x0,y0,x1,y1,x2,y2]';
% z_hat = [x0,y0]';

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