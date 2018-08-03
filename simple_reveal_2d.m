% restart
close all; clear all; clc;

% data storage
estError = [];

% initialize true state (x_t)
xt = [ 0.25 0.55 0.45 0.65 -pi/6 pi/2 ]';

% initialize state estimate (x) and covariance matrix (P)
x0 = [ 0.35 0.45 0.55 0.55 pi/12 pi/3 ]';
P0 = eye(length(x0));
x_prev = x0;
P_prev = P0;

% initialize state and measurement covariances
R = eye(length(x0)); % state noise covariance following Thrun, et al.
Q = eye(length(x0)); % measurement noise covariance following Thrun, et al.

% iterate through horizon levels
for revealLevel = 0:0.01:1.2
    
    % compute Jacobian of state transition function
    F = eye(length(x_prev));
    
    % prediction step: propigate dynamics forward in time
    x = x_prev;
    P = F*P_prev*F' + R;
    
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
    %     z = -1*ones(size(z_full));
    %     for pointIdx = 1:(length(z_full)/2)
    %         xIdx = 2*pointIdx - 1;
    %         yIdx = xIdx + 1;
    %        if( z_full(yIdx) < revealLevel )
    %            z(xIdx) = z_full(xIdx);
    %            z(yIdx) = z_full(yIdx);
    %        end
    %     end
    z_full = simpleRevealObsModel2d( xt );     % full observation is an error-free measurement of all point (x,y) values
    for pointIdx = 1:(length(z_full)/2)
        xIdx = 2*pointIdx - 1;
        yIdx = xIdx + 1;
        if( z_full(yIdx) < revealLevel )
            
            
            z_i =     [z_full(xIdx) z_full(yIdx)]';
            z_hat_full = simpleRevealObsModel2d( x );  % expected observation based on current model
            
            z_hat_i = [z_hat_full(xIdx) z_hat_full(yIdx)]';
            
            H_i = H([xIdx yIdx],:);
            Q_i = Q([xIdx yIdx],[xIdx yIdx]);
            
            K = P*H_i'*inv(H_i*P*H_i'+Q_i);
            x = x + K*(z_i - z_hat_i);
            P = (eye(size(K,1))-K*H_i)*P;
        end
    end
    
    % display current estimate and true state
    plotRevealModel2d( x, xt, revealLevel);
    
    rmse = computeReveal2dError(x,xt);
    disp(sprintf('RMSE = %f',rmse));
    estError(end+1,:) = [revealLevel rmse];
    
    figure(2);
    set(gcf,'Position',[0589 0225 0560 0420]);
    grid on;
    plot(estError(:,1),estError(:,2),'b-','LineWidth',1.6);
    xlabel('\bfHorizon Level');
    ylabel('\bfRMSE');
    xlim([0 1.5]);
    ylim([0 0.5]);
    
    % increment to next timestep
    x_prev = x;
    P_prev = P;
    
    pause(0.1);
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
function plotRevealModel2d( estParams, trueParams, horizonLevel )
colors = [0 0 1;1 0 0];
modelParams = {trueParams,estParams};
lineStyles = {'.-','.:'};
markerSizes = [75,60];

% switch to animation figure and turn hold off
figure(1);
set(gcf,'Position',[1.034000e+02 1.922000e+02 3.896000e+02 0508]);
hold off;
ploth = []; % storage for plot handles

% plot estimated and true models
for modelIdx = 1:length(modelParams)
    
    % get color settings and model parameters
    thisColor = colors(modelIdx,:);
    thisModel = modelParams{modelIdx};
    
    % convert model parameters to x,y locations of each point
    z = simpleRevealObsModel2d(thisModel);
    
    % plot a dummy point for correctly-scaled legend entry
    ploth(end+1) = plot(NaN,NaN,'.-','MarkerSize',25,'LineWidth',2,'Color',thisColor);
    hold on; grid on;
    
    % plot model
    plot(z([5 1 3]) ,z([6 2 4]),lineStyles{modelIdx},'MarkerSize',markerSizes(modelIdx),'LineWidth',2,'Color',thisColor);
    
end

% add occluded region
ploth(end+1) = patch([0 1 1 0], [horizonLevel horizonLevel 1.4 1.4],'k','FaceAlpha',0.2,'EdgeColor','none');

% add legend, scale axes, and draw plot
legh = legend(ploth,{'Estimate','Truth','Occlusion'},'Location','northoutside','Orientation','horizontal','FontWeight','bold');
ylim([0 1.4]);
axis equal;
xlim([0 1]);
axh = gca;
axh.XAxis.Visible = 'off';
axh.YAxis.Visible = 'off';
legh.Position = legh.Position + [0.02 0.04 0 0];
drawnow;

end

function rmse = computeReveal2dError(x,xt)

z_hat = simpleRevealObsModel2d(x);
z_t   = simpleRevealObsModel2d(xt);
z_d = z_t-z_hat;
rmse = sqrt( (z_d'*z_d)/(length(z_d)/2));

end