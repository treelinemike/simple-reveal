% restart
close all; clear all; clc;

% settings
maxHorizonLevel = 1.0;
doSaveFrames = 0;

% initialize frame count
frameCount = 0;

% data storage
estError = [];
obsCounts = zeros(3,1);

% initialize true state (x_t)
xt = [ 0.25 0.35 0.45 0.65 -pi/6 pi/2 ]';

% initialize state estimate (x) and covariance matrix (P)
% x0 = [ 0.35 0.25 0.55 0.55 pi/12 pi/3 ]';
% x0 = [ 0.80 0.45 0.35 0.35 -9*pi/8 3*pi/2 ]';
x0 = [ 0.80 0.25 0.5 0.25 3*pi/8 pi/4 ]';
P0 = 0.1*eye(length(x0));
x_prev = x0;
P_prev = P0;

% initialize state and measurement covariances
Q = 0.1*eye(length(x0)); % process noise covariance (note: Thrun, et al. calls this R!)
R = 0.1*eye(length(x0)); % measurement noise covariance (note: Thrun, et al. calls this Q!)

% iterate through horizon levels
for revealLevel = 0:0.01:maxHorizonLevel
    
    % compute Jacobian of state transition function
    F = eye(length(x_prev));
    
    % prediction step: propigate dynamics forward in time
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

        % generate observation and expected observation for this single point
        z_i =     [z_full(xIdx) z_full(yIdx)]';
        z_hat_full = simpleRevealObsModel2d( x );  % expected observation based on current model; this could probably be pre-computed for each timestep (don't recalculate between measurement updates at same timestep) - not sure whether results will differ        
        z_hat_i = [z_hat_full(xIdx) z_hat_full(yIdx)]';
        H_i = H([xIdx yIdx],:);
        R_i = R([xIdx yIdx],[xIdx yIdx]);
        
        % push model along y axis if 
        if( (z_hat_i(2) <= revealLevel) && (z_i(2) > revealLevel))
%                 z_i = [z_hat_i(1) revealLevel+(revealLevel-z_hat_i(2))];
%                 disp(['Trying to push model to ' sprintf('(%f,%f)',z_i(1),z_i(2)) ]);
%                 R_i = 50*R_i;
                
%                 delta_z = zeros(size(z_full));
%                 delta_z(yIdx) = revealLevel-z_hat_full(yIdx);
%                 x_synth = x + inv(H)*delta_z;
%                 x = x_synth;
                
        elseif((z_i(2) <= revealLevel) || (z_hat_i(2) <= revealLevel))
            
                
            
            % update state and error covariance
            K = P*H_i'*inv(H_i*P*H_i'+R_i);
            innov = (z_i - z_hat_i);
            innov2 = sign(innov).*(abs(innov).^1);
            x = x + K*innov2;
            P = (eye(size(K,1))-K*H_i)*P;
            
            % increment number of times this point has been seen for
            % attenuation
            if( (z_i(2) <= revealLevel) )
                obsCounts(pointIdx) =  obsCounts(pointIdx) + 1;
            end
        end
    end
    
    % display current estimate and true state
    plotRevealModel2d( x, xt, revealLevel);
    
    rmse = computeReveal2dError(x,xt);
    estError(end+1,:) = [revealLevel rmse];
    
    figure(1);
    subplot(4,1,4);
    plot(estError(:,1),estError(:,2),'b-','LineWidth',1.6);
    xlabel('\bfHorizon Level');
    ylabel('\bfRMSE');
    xlim([0 maxHorizonLevel]);
    ylim([0 0.65]);
    grid on;
    
    % save frames for animation
    % then convert - not in MATLAB although could use system() command - using a codec compatible with LaTeX (Beamer)
    % see: https://tex.stackexchange.com/questions/141622/trouble-using-ffmpeg-to-encode-a-h264-video-in-a-mp4-container-to-use-with-medi
    % first download FFMPEG for windows: https://ffmpeg.zeranoe.com/builds/ OR https://www.ffmpeg.org/download.html
    % ffmpeg -r 10 -start_number 1 -i frame%003d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4
    
    if(doSaveFrames)
        saveas(gcf,sprintf('frame%03d.png',frameCount));
        frameCount = frameCount + 1;
    end
    
    
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
% set(gcf,'Position',[1.034000e+02 1.922000e+02 3.896000e+02 5.080000e+02]);
% set(gcf,'Position',[1.034000e+02 0069 3.896000e+02 6.984000e+02]);
set(gcf,'Position',[1.034000e+02 2.274000e+02 0356 0540]);
subplot(4,1,1:3);

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
legh = legend(ploth,{'Truth','Estimate','Occlusion'},'Location','northoutside','Orientation','horizontal','FontWeight','bold');
ylim([0 1.2]);
axis equal;
xlim([0 1]);
axh = gca;
axh.XAxis.Visible = 'off';
axh.YAxis.Visible = 'off';
legh.Position = legh.Position + [0.08 0.04 0 0];
drawnow;

end

function rmse = computeReveal2dError(x,xt)

z_hat = simpleRevealObsModel2d(x);
z_t   = simpleRevealObsModel2d(xt);
z_d = z_t-z_hat;
rmse = sqrt( (z_d'*z_d)/(length(z_d)/2));

end