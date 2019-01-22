% particle filter test

% restart
close all; clear all; clc;
% global flag for plot initialization
global plotCallCount gh_patch gh_truth gh_est gh_rmse gh_truth_dot gh_est_dot revealLevels;
plotCallCount = 0;
revealLevels = 0:0.01:1;
doRegularize = 1;

M = 4^6; % number of particles

% initialize true state (x_t)
xt = [ 0.25 0.35 ]; % 0.45 0.65 -pi/6 pi/2 ]';

N = length(xt);  % number of states

% initialize state estimate (x) and covariance matrix (P)
% x0 = [ 0.35 0.25 0.55 0.55 pi/12 pi/3 ]';

%x0 = xt + [0.02 -0.03];% -0.05 -0.07 2*pi/180 -4*pi/180]';
x0 = [1 1];

% x0 = [ 0.80 0.45 0.35 0.35 -9*pi/8 3*pi/2 ]';
% x0 = [ 0.80 0.25 0.5 0.25 3*pi/8 pi/4 ]';
% P0 = diag([2 2 2 2 pi pi]); %1*ones(length(x0));%0.1*eye(length(x0));
% P0 = diag([0.5 0.5 0.5 0.5 15*pi/180 15*pi/180]); %1*ones(length(x0));%0.1*eye(length(x0));
P0 = diag([0.2 0.2]);% 0.2 0.2 4*pi/180 4*pi/180]);
x = x0;

P = P0;
% R = diag([0.05 0.05 0.05 0.05 0.05 0.05]); % measurement noise covariance (note: Thrun, et al. calls this Q!)

% P = eye(6);
R = 0.05*eye(2);%6);

X = mvnrnd(x,P,M)';
% ksdensity(X(1,:)); % show marginal PDFs


revealLevel = 0.95;  % ideally we'd iterate this, but we'll keep it static for this test

plotRevealModel2d( x, xt, P, revealLevel);

% wait
pause(0.5);


for i = 1:50
    q = zeros(M,1);
    q2 = q;
    X_prior = zeros(N,M);
    
    figure(2);
    hold off; 
    plot(X(1,:),X(2,:),'b.','MarkerSize',20); 
    xlim([0 2]); 
    ylim([0 2]);
    hold on;
    plot(xt(1),xt(2),'ro','MarkerSize',10,'LineWidth',2);
    plot(x0(1),x0(2),'k.','MarkerSize',20,'LineWidth',2);

    
    
    for m = 1:M
        
        % get the appropriate particle
        x_prev = X(:,m);
        
        % compute proposal (prediction) for this particle
        x_prior = x_prev;  % identity transformation
        X_prior(:,m) = x_prior;
        
        % get a measurement
        z = simpleRevealObsModel2d( xt );
        z_hat = simpleRevealObsModel2d( x_prior );
        innov = z-z_hat;
        
        % remove unobserved rows from innovation and R matrix
        obs_mask = repelem(z(2:2:end,:) < revealLevel,2);
        innov = innov(obs_mask);
        Rt = R(obs_mask,obs_mask);
        
        % compute importance factor
        c1 = (1/( (2*pi)^(size(innov,1)/2) * sqrt(det(Rt)) ));
        q(m) = c1 * exp( -0.5 * innov' * inv(Rt) * innov);  % the q vector incorporates MEASUREMENT data
        q2(m) = exp( -0.5 * innov' * inv(Rt) * innov);  
        
    end
    q = q/sum(q);
    
    % Do we really need to use the coefficient of the exponential? Probably
    % not...
    q2 = q2/sum(q2);
    
    % max(abs(q - q2))
    %     figure;
    %     hold on; grid on;
    %     plot(cumsum(q),'r');
    %     plot(cumsum(q2),'b--');
    %
    qCDF = cumsum(q);
    
    
    if(doRegularize)
        %    Simon pg. 473: regularized particle filtering
        
        % compute particle ensemble mean and covariance matrix
        
        x_post_idx = arrayfun(@(afin1) find(afin1 < qCDF,1,'first'), rand(M,1)-eps );  % eps subtracted from rand(M,1) to exclude exactly 1.0000... though this is extremely unlikely
        X_post_prereg = X_prior(:,x_post_idx);
        
        % mean 
        mu = mean(X_post_prereg,2); % equivalent to: (1/M)*sum(X_prior,2);
        
        % variance
        S = zeros(N);
        for mIdx = 1:M
            S = S + (X_post_prereg(:,mIdx)-mu)*(X_post_prereg(:,mIdx)-mu)';
        end
        S = (1/(M-1))*S;

        X_post = mvnrnd(mu,1.0*S,M)';
        x_post = mean(X_post,2);

        
        figure(2)
        plot(x_post(1),x_post(2),'m*','MarkerSize',20);


        

        
%         % factor covariance matrix as S = A*A' (Cholesky)... note, need 'lower'
%         % argument in MATLAB
%         A = chol(S,'lower');
%         
%         % compute volume of n-dim unit sphere
%         vn = zeros(1,N);
%         vn(1) = 2;
%         vn(2) = pi;
%         for vnIdx = 3:N
%             vn(vnIdx) = 2*pi*vn(vnIdx-2)/vnIdx;
%         end
%         vn = vn(end);
%         
        % compute optimal kernel bandwidth
%         h = 0.5 * ((8*(1/vn)*(N+4)*(2*sqrt(pi))^N)^(1/(N+4))) * (M)^(-1/(N+4));
        
        % approximate posterior PDF
        % define state space bounds and sampling density
        % randomly sample PDF 'near' a priori estimate: A = mu + randn(6,1000).*sqrt(diag(S));  [mu mean(A,2) sqrt(diag(S)) std(A')']
        % or maybe with a beta? mu = .6; y = betarnd(ones(100,1)*2,ones(100,1)*2*(1-mu)/mu); close all; [ksy,ksx] = ksdensity(y); plot(ksx,ksy); [~,maxIdx] = max(ksy); [mu mean(y) ksx(maxIdx)]
        % better yet: truncated normal: pd = makedist('normal','mu',0,'sigma',2); pd2 = truncate(pd,-1,1); x = -10:0.01:10; y = pdf(pd,x); close all; plot(x,y,'b');hold on; plot(x,pdf(pd2,x),'r')
        %
%         numPtsPerDim = 4;  % critical parameter, size of sample point list is numPtsPerDim^N where N is the number of dimensions
%         stdevs = sqrt(diag(S));
%         samplePtsPerDim = zeros(N,numPtsPerDim);
%         bounds = [0 1; 0 1];%; 0 1; 0 1; -pi pi; -pi pi];
%         for dimIdx = 1:N
%             samplePtsPerDim(dimIdx,:) = ...
%                 random( ...
%                 truncate( ...
%                 makedist('normal','mu',mu(dimIdx),'sigma',stdevs(dimIdx)) ...
%                 ,bounds(dimIdx,1),bounds(dimIdx,2)) ...
%                 ,[1,numPtsPerDim]);
%         end
%         
%         [X1 X2]  = ndgrid(...  %X3 X4 X5 X6]
%             samplePtsPerDim(1,:), ...
%             samplePtsPerDim(2,:) );%, ...
%             %samplePtsPerDim(3,:), ...
%             %samplePtsPerDim(4,:), ...
%             %samplePtsPerDim(5,:), ...
%             %samplePtsPerDim(6,:) );
%         samplePts = [X1(:), X2(:)]';%, X3(:) X4(:) X5(:) X6(:)]';         % list of state space locations at which to sample posterior
%         
%         posteriorVals = zeros(size(samplePts,2),1);
%         normcheck.yes = 0;
%         normcheck.no = 0;
%        
%         for sampIdx = 1:size(samplePts,2)
%             xs = samplePts(:,sampIdx);
%             thisPosteriorVal = 0;
%             for particleIdx = 1:M
%                 x = inv(A)*((xs-X_prior(:,particleIdx))/h);
%                 
%                 kernelCov = eye(size(x,1));
%                 c2 = (1/( (2*pi)^(size(x,1)/2) * sqrt(det(kernelCov)) ));
%                 Kx = c2 * exp( -0.5 * x' * inv(kernelCov) * x);  % the q vector incorporates MEASUREMENT data
%         
% %                 Kx = (1/sqrt(2*pi)*1)*exp(-0.5*norm(x)^2/(1^2));  % Gaussian kernel
%                 Khx = (1/det(A))*(1/(h^N))*Kx;
%                 thisPosteriorVal = thisPosteriorVal + q(particleIdx)*Khx;
%             end
%             posteriorVals(sampIdx) = thisPosteriorVal;
%         end
% 
%         % normalize posterior values
%         posteriorVals = posteriorVals/sum(posteriorVals);
%         posteriorCDFVals = cumsum(posteriorVals);
        

        % estimate the mean of this PDF ...
        % alternatively we could do this from sampled particle set
%         x_post = sum(samplePts.*repmat(posteriorVals',N,1),2);
        
        % sample new particle set
%         randVec = rand(M,1)-eps;
%         x_post_idx = arrayfun(@(afin1) find(posteriorCDFVals > afin1,1,'first'), randVec );
%         X_post = samplePts(:,x_post_idx);
        
    else
        %    Simon pg 468 step 3d:
        x_post_idx = arrayfun(@(afin1) find(afin1 < qCDF,1,'first'), rand(M,1)-eps );  % eps subtracted from rand(M,1) to exclude exactly 1.0000... though this is extremely unlikely
        X_post = X_prior(:,x_post_idx);
        x_post = sum(X_prior.*repmat(q',N,1),2);
        
    end
    % % % %     % MAP estimate -- probably wrong! should NOT do this on marginal
    % % % %     % distributions, right? Better to find maximum value of PDF over entire
    % % % %     % state space?
    % % % %     x_post = zeros(size(x_prior));
    % % % %     for xIdx = 1:length(x_post)
    % % % %         [pdfY pdfX] = ksdensity(X_post(xIdx,:));
    % % % %         [~,mapIdx] = max(pdfY);
    % % % %         x_post(xIdx) = pdfX(mapIdx);
    % % % %     end
    % % %
    % % %     % posterior mean
    % % %     %     x_post = mean(X_post,2);
    
    % display result
    plotRevealModel2d( x_post, xt, P, revealLevel);
    
    % wait
    pause(0.1);
    
    % update particle set
    X = X_post;
end
% convert model parameters into (x,y) locations of each point

function z_hat = simpleRevealObsModel2d( x )

x0 = x(1);
y0 = x(2);
% x1 = x0 + x(3)*cos(x(5));
% y1 = y0 + x(3)*sin(x(5));
% x2 = x0 + x(4)*cos(x(5)+x(6));
% y2 = y0 + x(4)*sin(x(5)+x(6));

% z_hat = [x0,y0,x1,y1,x2,y2]';
z_hat = [x0,y0]';

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
    gh_truth = line(z_truth(1),z_truth(2),'Marker','.','MarkerSize',75,'LineStyle','-','Color',[0 0 1],'LineWidth',2.0);
%     gh_truth_dot = line(z_truth(5),z_truth(6),'Marker','.','MarkerSize',25,'Color',[1 1 1]);
    gh_est = line(z_est(1),z_est(2),'Marker','.','MarkerSize',60,'LineStyle',':','Color',[1 0 0],'LineWidth',2.0);
%     gh_est_dot = line(z_est(5),z_est(6),'Marker','.','MarkerSize',25,'Color',[1 1 1]);
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
    gh_truth.XData = z_truth(1);
    gh_truth.YData = z_truth(2);
%     gh_truth_dot.XData = z_truth(5);
%     gh_truth_dot.YData = z_truth(6);
    gh_est.XData = z_est(1);
    gh_est.YData = z_est(2);
%     gh_est_dot.XData = z_est(5);
%     gh_est_dot.YData = z_est(6);
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