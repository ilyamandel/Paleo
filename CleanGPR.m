%%%% Data capture -- this should be updated to point to the relevant files
%%%% the expected file format contains the following columns in order: 
%%%% modern file: ID GDGT0 GDGT1 GDGT2 GDGT3 Cren Cren' long lat Temp Depth
%%%% ancient file: ID GDGT0 GDGT1 GDGT2 GDGT3 Cren Cren'

modern=xlsread('~/Work/Paleo/Data201803.xlsx','Modern Calibration');
%ancient=xlsread('~/Work/Paleo/Data201803.xlsx','Cretaceous');
ancient=xlsread('~/Work/Paleo/OPTiMAL_Eocene_new.xlsx'); ancient=ancient(:,2:end);

%calibrate GP regression on full modern data set
gprMdl = fitrgp(modern(:,1:6),modern(:,9),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',std(modern(:,[1:6,9])),'Sigma',std(modern(:,9)));
gprMdl.KernelInformation.KernelParameters./(std(modern(:,[1:6,9])))'    
[tempmodern,tempmodernstd,tempmodern95]=predict(gprMdl,modern(:,1:6));
L=loss(gprMdl,modern(:,1:6),modern(:,9));
sqrt(L)
std(modern(:,9)-tempmodern)
sigmaL = gprMdl.KernelInformation.KernelParameters(1:end-1); % Learned length scales
weights = exp(-sigmaL); % Predictor weights
weights = weights/sum(weights)

%apply GP regression to ancient data set
[tempancient,tempancientstd,tempancient95]=predict(gprMdl,ancient(:,1:6),'Alpha',0.05);

%determine weighted nearest neighbour distances
for(i=1:length(ancient)),
    for(j=1:length(modern)),
            dist=(modern(j,1:6)-ancient(i,1:6))./sigmaL';
            distsq(j)=sqrt(sum(dist.^2));
    end;
    [distmin(i),index(i)]=min(distsq);
end;

figure(18), set(gca, 'FontSize', 16); 
scatter(log(distmin)/log(10),tempancientstd, 'filled'); hold on;
plot(log(0.5)/log(10)*ones(size([3:9])),[3:9],'k:'); hold off;
set(gca, 'FontSize', 24); 
xlabel('$\log_{10}(D_\mathrm{nearest,weighted})$','Interpreter', 'latex'),
ylabel('Ancient prediction standard deviation')




%%%%%%
%Test for Sarah
figure(19), set(gca, 'FontSize', 16);
scatter(tempancient,ancient(:,8), 'filled');
set(gca, 'FontSize', 24); 
xlabel('T_{GPR} according to Ilya'),
ylabel('T_{GPR} according to Will')

figure(20), set(gca, 'FontSize', 16);
scatter(log(distmin)/log(10),log(ancient(:,7))/log(10), 'filled');
set(gca, 'FontSize', 24); 
xlabel('$\log_{10}(D_\mathrm{nearest,weighted})$, Ilya','Interpreter', 'latex'),
ylabel('$\log_{10}(D_\mathrm{nearest,weighted})$, Will','Interpreter', 'latex');

figure(21), set(gca, 'FontSize', 16);
scatter(tempancient(distmin<0.5),ancient(distmin<0.5,10), 25, log(distmin(distmin<0.5))/log(10), 'filled'); hold on;
c = colorbar;
scatter(tempancient(distmin>=0.5),ancient(distmin>=0.5,10), 15, [0.5 0.5 0.5], 'filled');
set(gca, 'FontSize', 24); 
xlabel('T_{GPR} according to Ilya'),
ylabel('T_{forward} according to Will');
c.Label.String = 'log_{10}(D_{nearest}), Ilya';
plot(tempancient,tempancient, 'k:'); hold off;

figure(22), set(gca, 'FontSize', 16);
scatter(tempancient(distmin<0.5),ancient(distmin<0.5,10), 25, tempancientstd(distmin<0.5), 'filled'); hold on;
c = colorbar;
scatter(tempancient(distmin>=0.5),ancient(distmin>=0.5,10), 15, [0.5 0.5 0.5], 'filled');
set(gca, 'FontSize', 24); 
xlabel('T_{GPR} according to Ilya'),
ylabel('T_{forward} according to Will');
c.Label.String = 'Temperature uncertainty, Ilya';
plot(tempancient,tempancient, 'k:'); hold off;


%for Yvette
%ancient=xlsread('~/Work/Paleo/YEMarylandMiocene.xlsx');
ancient=xlsread('~/Work/Paleo/JamesSuper2018Geologydata.xlsx');  ancient=ancient(:,11:16);
[tempancient,tempancientstd,tempancient95]=predict(gprMdl,ancient(:,1:6),'Alpha',0.05);

%determine weighted nearest neighbour distances
for(i=1:length(ancient)),
    for(j=1:length(modern)),
            dist=(modern(j,1:6)-ancient(i,1:6))./sigmaL';
            distsq(j)=sqrt(sum(dist.^2));
    end;
    [distmin(i),index(i)]=min(distsq);
end;

figure(18), set(gca, 'FontSize', 16); 
scatter(log(distmin)/log(10),tempancientstd, 'filled'); hold on;
plot(log(0.5)/log(10)*ones(size([3:9])),[3:9],'k:'); hold off;
set(gca, 'FontSize', 24); 
xlabel('$\log_{10}(D_\mathrm{nearest,weighted})$','Interpreter', 'latex'),
ylabel('Ancient prediction standard deviation')

[tempancient';tempancientstd';distmin]'

