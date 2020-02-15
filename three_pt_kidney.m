function kidney = three_pt_kidney(zt,R_param)
% % reset
% close all; clear; clc;
% 
% xt = [ 0.25 0.35 0.65 0.45 -pi/3 17*pi/12]';
% zt = simpleRevealObsModel2d(xt);
% figure;
% hold on; grid on;
% axis equal;
% 
% plot(zt(1),zt(2),'r.','MarkerSize',20);
% plot(zt(3),zt(4),'g.','MarkerSize',20);
% plot(zt(5),zt(6),'b.','MarkerSize',20);
% 
% R_param = 0.2;

v1 = [zt(5:6);0]-[zt(1:2);0];
v2 = [zt(3:4);0]-[zt(1:2);0];
v3 = cross(v1,v2);

% plot(zt(1) + [0 v1(1)],zt(2) + [0 v1(2)],'b--');
% plot(zt(1) + [0 v2(1)],zt(2) + [0 v2(2)],'g--');


if(v3(3) < 0)
    % concave right
    
    arm1 = R_param*cross([0;0;1],unitvec(v1));
    arm2 = R_param*cross([0;0;-1],unitvec(v2));
%     plot( zt(5) + [0 arm1(1)], zt(6) + [0 arm1(2)],'b--');
%     plot( zt(3) + [0 arm2(1)], zt(4) + [0 arm2(2)],'g--');
    gamma = acos(dot(arm1,arm2)/(norm(arm1)*norm(arm2))); % correct
    start_ang = acos(dot([-1;0;0],arm2)/(norm(arm2)));
    
    theta_end = pi+start_ang;
    theta_start = theta_end -gamma;
    theta = (theta_start:0.001:theta_end)';
    theta_u = theta_start + gamma/2 -pi;
    u_hat = [cos(theta_u);sin(theta_u);0];
    
%     plot(zt(1)+[0 u_hat(1)],zt(2)+[0 u_hat(2)],'r--');
    
    if( norm(v1) < norm(v2) )
        l = norm(v1)/(dot(u_hat,unitvec(v1)));
        ctr = [zt(1:2);0]+l*u_hat;
%         plot(zt(1)+[0 l*u_hat(1)],zt(2)+[0 l*u_hat(2)],'b.','MarkerSize',20)
        R = norm([zt(5:6);0]-ctr)+R_param;
%         genericLine = (norm(v2)-norm(v1))*unitvec(v2);
%         outLine = ctr(1:2)' + R*[cos(theta_end) sin(theta_end)] + [0 0; genericLine(1:2)'];
%         inLine = ctr(1:2)' + (R-2*R_param)*[cos(theta_end) sin(theta_end)] + [0 0; genericLine(1:2)'];
    else
        l = norm(v2)/(dot(u_hat,unitvec(v2)));
        ctr = [zt(1:2);0]+l*u_hat;
%         plot(zt(1)+[0 l*u_hat(1)],zt(2)+[0 l*u_hat(2)],'b.','MarkerSize',20)
        R = norm([zt(3:4);0]-ctr)+R_param;
%         genericLine = (norm(v1)-norm(v2))*unitvec(v1);
%         outLine = ctr(1:2)' + R*[cos(theta_start) sin(theta_start)] + [0 0; genericLine(1:2)'];
%         inLine = ctr(1:2)' + (R-2*R_param)*[cos(theta_start) sin(theta_start)] + [0 0; genericLine(1:2)'];
        
    end
%     plot(ctr(1),ctr(2),'b.','MarkerSize',20);
%     plot(outLine(:,1),outLine(:,2),'m--');
%     plot(inLine(:,1),inLine(:,2),'m--');
    
    reararc = R*[cos(theta) sin(theta)]+ctr(1:2)';
    frontarc = (R-2*R_param)*[cos(theta) sin(theta)]+ctr(1:2)';
%     plot(reararc(:,1),reararc(:,2),'m--');
%     plot(frontarc(:,1),frontarc(:,2),'m--');
    
    theta_end_2 = theta_start;
    theta_start_2 = theta_end_2-pi;
    theta_2 = (theta_start_2:0.001:theta_end_2)';
    toparc = R_param*[cos(theta_2) sin(theta_2)]+zt(5:6)';
%     plot(toparc(:,1),toparc(:,2),'m--');
    
    theta_start_3 = theta_end;
    theta_end_3 = theta_start_3+pi;
    theta_3 = (theta_start_3:0.001:theta_end_3)';
    botarc = R_param*[cos(theta_3) sin(theta_3)]+zt(3:4)';
%     plot(botarc(:,1),botarc(:,2),'m--');
    
    kidney = [reararc;botarc;flipud(frontarc);toparc];
else
    % concave left
    
    arm1 = R_param*cross([0;0;-1],unitvec(v1));
    arm2 = R_param*cross([0;0;1],unitvec(v2));
%     plot( zt(5) + [0 arm1(1)], zt(6) + [0 arm1(2)],'b--');
%     plot( zt(3) + [0 arm2(1)], zt(4) + [0 arm2(2)],'g--');
    gamma = acos(dot(v1,v2)/(norm(v1)*norm(v2))); % correct
    start_ang = acos(dot([1;0;0],arm2)/(norm(arm2)));
    
    theta_start =start_ang;
    theta_end = start_ang + pi-gamma;
    theta = (theta_start:0.001:theta_end)';
    theta_u = (theta_start + theta_end)/2 + pi; 
    u_hat = [cos(theta_u);sin(theta_u);0];
    
%     plot(zt(1)+[0 u_hat(1)],zt(2)+[0 u_hat(2)],'r--');
    
    if( norm(v1) < norm(v2) )
        l = norm(v1)/(dot(u_hat,unitvec(v1)));
        ctr = [zt(1:2);0]+l*u_hat;
%         plot(zt(1)+[0 l*u_hat(1)],zt(2)+[0 l*u_hat(2)],'b.','MarkerSize',20)
        R = norm([zt(5:6);0]-ctr)+R_param;
    else
        l = norm(v2)/(dot(u_hat,unitvec(v2)));
        ctr = [zt(1:2);0]+l*u_hat;
%         plot(zt(1)+[0 l*u_hat(1)],zt(2)+[0 l*u_hat(2)],'b.','MarkerSize',20)
        R = norm([zt(3:4);0]-ctr)+R_param;
    end
%     plot(ctr(1),ctr(2),'b.','MarkerSize',20);
%     plot(outLine(:,1),outLine(:,2),'m--');
%     plot(inLine(:,1),inLine(:,2),'m--');
    
    reararc = R*[cos(theta) sin(theta)]+ctr(1:2)';
    frontarc = (R-2*R_param)*[cos(theta) sin(theta)]+ctr(1:2)';
%     plot(reararc(:,1),reararc(:,2),'m--');
%     plot(frontarc(:,1),frontarc(:,2),'m--');
    
    theta_end_2 = theta_end + pi;
    theta_start_2 = theta_end;
    theta_2 = (theta_start_2:0.001:theta_end_2)';
    toparc = R_param*[cos(theta_2) sin(theta_2)]+zt(5:6)';
%     plot(toparc(:,1),toparc(:,2),'m--');
    
    theta_start_3 = theta_start-pi;
    theta_end_3 = theta_start;
    theta_3 = (theta_start_3:0.001:theta_end_3)';
    botarc = R_param*[cos(theta_3) sin(theta_3)]+zt(3:4)';
%     plot(botarc(:,1),botarc(:,2),'m--');

    kidney = [reararc;toparc;flipud(frontarc);botarc];
end
