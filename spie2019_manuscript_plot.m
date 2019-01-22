% restart
close all; clear all; clc;

revealLevels = 0:0.01:1.0;
xt = [ 0.25 0.35 0.45 0.65 -pi/6 pi/2 ]';
ic1 = [ 0.35 0.25 0.55 0.55 pi/12 pi/3 ]';
ic2 = [ 0.80 0.45 0.35 0.35 -9*pi/8 3*pi/2 ]';
ic3 = [ 0.80 0.25 0.5 0.25 3*pi/8 pi/4 ]';


ekf1 = load('rmse_ekf_noproj_ic1.mat');
ekf2 = load('rmse_ekf_noproj_ic2.mat');
ekf3 = load('rmse_ekf_noproj_ic3.mat');

ekfp1 = load('rmse_ekf_proj_ic1.mat');
ekfp2 = load('rmse_ekf_proj_ic2.mat');
ekfp3 = load('rmse_ekf_proj_ic3.mat');

eif1 = load('rmse_eif_noproj_ic1.mat');
eif2 = load('rmse_eif_noproj_ic2.mat');
eif3 = load('rmse_eif_noproj_ic3.mat');
 
pf1 = load('rmse_pf_noproj_ic1.mat');
pf2 = load('rmse_pf_noproj_ic2.mat');
pf3 = load('rmse_pf_noproj_ic3.mat');
 
icData = {};
icData{end+1} = [revealLevels' ekf1.rmse' ekfp1.rmse' eif1.rmse' pf1.rmse'];
icData{end+1} = [revealLevels' ekf2.rmse' ekfp2.rmse' eif2.rmse' pf2.rmse'];
icData{end+1} = [revealLevels' ekf3.rmse' ekfp3.rmse' eif3.rmse' pf3.rmse'];

ax = [];
plotLineStyles = {'-','--',':','-.'};
figure;
set(gcf,'Position',[0005 3.418000e+02 1528 2.584000e+02]);

for icIdx = 1:length(icData)
    
    thisData = icData{icIdx};
    
    ax(end+1) = subplot(1,length(icData),icIdx);
    hold on; grid on;
    for traceIdx = 1:4
        plot(thisData(:,1),thisData(:,1+traceIdx),'LineWidth',2,'LineStyle',plotLineStyles{traceIdx});
    end
    xlabel('\bfHorizon Level');
    ylabel('\bfRMSE');
    ylim([0 1]);
end

linkaxes(ax,'y');

subplot(1,length(icData),1);
legend('EKF','EKF w/ Projection','EIF','PF','Location','northwest');


plotRevealModel2d(ic1,xt);
plotRevealModel2d(ic2,xt);
plotRevealModel2d(ic3,xt);

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

function plotRevealModel2d( estParams, trueParams )

% plotting settings, etc.
colors = [0 0 1;1 0 0];
modelParams = {trueParams,estParams};
lineStyles = {'.-','.:'};
markerSizes = [75,60];
ploth = [];  % storage for plot handles

% switch to animation figure and turn hold off
figure;
hold off;

% plot estimated and true models
for modelIdx = 1:length(modelParams)
    
    % get color settings and model parameters
    thisColor = colors(modelIdx,:);
    thisModel = modelParams{modelIdx};
    
    % convert model parameters to x,y locations of each point
    z = simpleRevealObsModel2d(thisModel);
    
    % plot a dummy point for correctly-scaled legend entry
    ploth(end+1) = plot(NaN,NaN,'.-','MarkerSize',25,'LineWidth',2,'Color',thisColor);
    hold on;
    
    % plot model
    plot(z([5 1 3]) ,z([6 2 4]),lineStyles{modelIdx},'MarkerSize',markerSizes(modelIdx),'LineWidth',2,'Color',thisColor);
    
    % add a point to one end for orientation purposes
    plot(z(5),z(6),'.','MarkerSize',20,'Color','w');
    
end

ylim([0 1.2]);
axis equal;
xlim([0 1.2]);
axh = gca;
axh.XAxis.Visible = 'off';
axh.YAxis.Visible = 'off';

end
