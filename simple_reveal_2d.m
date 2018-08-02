% restart
close all; clear all; clc;

% initialize true state and estimate
xt = [ 0.20 0.40 0.40 0.60 -pi/6 pi/2 ]';
x0 = [ 0.30 0.40 0.50 0.50 pi/12 pi/3 ]';

% initialize covariance matrix
P = eye(length(x0));

% iterate through horizon levels
for revealLevel = 0:0.05:1.2
    
    plotRevealModel2d( x0, xt, revealLevel);
%     pause(0.1);
end

function z = simpleRevealObsModel2d( x )

x0 = x(1);
y0 = x(2);
x1 = x0 + x(3)*cos(x(5));
y1 = y0 + x(3)*sin(x(5));
x2 = x0 + x(4)*cos(x(5)+x(6));
y2 = y0 + x(4)*sin(x(5)+x(6));

z = [x0,y0,x1,y1,x2,y2]';

end

function plotRevealModel2d( estParams, trueParams, horizonLevel )
colors = [1 0 0; 0 0 1];
modelParams = {estParams,trueParams};

ploth = [];

figure(1);
set(gcf,'Position',[1.034000e+02 1.922000e+02 3.896000e+02 0508]);
hold off;

for modelIdx = 1:length(modelParams)
    thisColor = colors(modelIdx,:);
    thisModel = modelParams{modelIdx};
    z = simpleRevealObsModel2d(thisModel);
    
    ploth(end+1) = plot(NaN,NaN,'.-','MarkerSize',25,'LineWidth',2,'Color',thisColor);
    hold on; grid on;
    
    plot(z([5 1 3]) ,z([6 2 4]),'.-','MarkerSize',75,'LineWidth',2,'Color',thisColor);
    
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