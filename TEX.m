cd ~/Work/Paleo
data=load('TEX.dat');
%Satellite T (°C)	GDGT-0	GDGT-1	GDGT-2	GDGT-3	Cren(archeol)	Cren'(isomer)	
T=data(:,1);
GDGT1=log(data(:,4)./(data(:,3)+data(:,4)+data(:,5)))/log(10);
TEX86=(data(:,4)+data(:,5)+data(:,7))./(data(:,3)+data(:,4)+data(:,5)+data(:,7));
TEXH=38.6+68.4*log(TEX86)/log(10);
OneTEX=54.5-19.1./TEX86;
TEXL=67.5*GDGT1+46.9;

plot(T,[TEXH';OneTEX';TEXL']','.'), %axis([-5 30 -30 30])
set(gca, 'FontSize', 20)
xlabel('Observed temperature'), ylabel('Predicted temperature'),
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Location', 'SouthEast');

plotregression(T',real(TEXH)')

%follow example at http://www.mathworks.com/help/nnet/examples/house-price-estimation.html?prodcode=NN&language=en
t=T';
x=data(:,2:7)';
net = fitnet(15);
view(net)
[net,tr] = train(net,x,t);
nntraintool
plotperform(tr)

trainX = x(:,tr.trainInd);
trainT = t(:,tr.trainInd);
trainY = net(trainX);

y = net(x);
plot(t,y,'o'); hold on; plot(trainT,trainY,'.r'); hold off;
set(gca, 'FontSize', 20)
xlabel('Observed temperature'), ylabel('Predicted temperature')

plot(T,[TEXH';OneTEX';TEXL';y]','.', 'MarkerSize', 10), %axis([-5 30 -30 30])
set(gca, 'FontSize', 20)
xlabel('Observed temperature'), ylabel('Predicted temperature'),
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Neural net', 'Location', 'SouthEast');
hold on; plot(T,T,'k--'); hold off;
plot(T(1:70),[TEXH(1:70)';OneTEX(1:70)';TEXL(1:70)';y(1:70)]','d', 'MarkerSize', 10), %axis([-5 30 -30 30])
hold off;

plot(T,[TEXH';OneTEX';TEXL';y]','.', 'MarkerSize', 10), %axis([-5 30 -30 30])
set(gca, 'FontSize', 20)
xlabel('Observed temperature'), ylabel('Predicted temperature'),
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Neural net', 'Location', 'SouthEast');
hold on; plot(T,T,'k--'); hold off;

plot(T(1:70),[TEXH(1:70)';OneTEX(1:70)';TEXL(1:70)';y(1:70)]','d', 'MarkerSize', 10), %axis([-5 30 -30 30])
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Neural net', 'Location', 'SouthEast');
hold on; plot(T,T,'k--'); hold off;


Rockall=load('Rockall.dat');
GDGT1=log(Rockall(:,4)./(Rockall(:,3)+Rockall(:,4)+Rockall(:,5)))/log(10);
TEX86=(Rockall(:,4)+Rockall(:,5)+Rockall(:,7))./(Rockall(:,3)+Rockall(:,4)+Rockall(:,5)+Rockall(:,7));
TEXH=38.6+68.4*log(TEX86)/log(10);
OneTEX=54.5-19.1./TEX86;
TEXL=67.5*GDGT1+46.9;

for(i=1:length(Rockall)),
    RockallX(:,i)=Rockall(i,2:7)'/sum(Rockall(i,2:7)); %normalize
end;
RockallDepth=Rockall(:,1)';
plot(RockallDepth,[TEXH';OneTEX';TEXL';net(RockallX)]'),
set(gca, 'FontSize', 20)
xlabel('Rockall depth'), ylabel('Predicted temperature')
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Neural net', 'Location', 'NorthEast');


net10 = fitnet(15,'trainbr' );%fitnet(10);
[net10,tr] = train(net10,x,t);
y=net10(x);
perform(net10,y,t)

net10b = fitnet(15,'trainbr' );%fitnet(10,'trainbr' );
[net10b,tr] = train(net10b,x,t);
y=net10b(x);
perform(net10b,y,t)

net10c = fitnet(15,'trainbr' );%fitnet(10,'traincgb' );
[net10c,tr] = train(net10c,x,t);
y=net10c(x);
perform(net10c,y,t)

net20 = fitnet(15,'trainbr' );%fitnet(20);
[net20,tr] = train(net20,x,t);
y=net20(x);
perform(net20,y,t)

net20b = fitnet(15,'trainbr' );%fitnet(20, 'trainbr' );
[net20b,tr] = train(net20b,x,t);
y=net20b(x);
perform(net20b,y,t)



for(i=1:length(Rockall)),
    RockallX(:,i)=Rockall(i,2:7)'/sum(Rockall(i,2:7)); %normalize
end;
RockallDepth=Rockall(:,1)';
nn=[net10(RockallX);net10b(RockallX);net10c(RockallX); net20(RockallX); net20b(RockallX)];
%plot(RockallDepth,nn); legend('N1', 'N2','N3', 'N4', 'N5', 'Location', 'NorthEast');
plot(RockallDepth,[TEXH';OneTEX';TEXL';mean(nn)]')%,'.', 'MarkerSize', 10), 
axis([95 136 10 40])
legend('TEX_{86}^H', '1/TEX_{86}','TEX_{86}^L', 'Neural net', 'Location', 'SouthEast');
%plot(RockallDepth,[net10(RockallX);net10b(RockallX);net10c(RockallX); net20(RockallX); net20b(RockallX)]),
set(gca, 'FontSize', 20)
xlabel('Rockall depth'), ylabel('Predicted temperature')
mnn=mean(nn);snn=std(nn);mean(mnn(5:10)), 1/sqrt(mean(1./snn(5:10).^2))/sqrt(6)

cd ~/Work/Paleo
data=load('TEX.dat');
stddata=std(data);
Rockall=load('Rockall.dat');
for(i=1:length(Rockall)),
    RockallX(:,i)=Rockall(i,2:7)'/sum(Rockall(i,2:7)); %normalize
end;
for(i=1:length(Rockall)),
    for(j=1:length(data)),
        dist=((data(j,2:7)-RockallX(:,i)')./stddata(2:7));
        distsq(j)=sqrt(sum(dist.^2));
    end;
    distmin(i)=min(distsq);
end;
hist(distmin,100); set(gca, 'FontSize', 20); 
xlabel('Normalized distance to closest calibration point')




%%%%%%%%%%%%%%%%%%%%%%

modern=xlsread('~/Work/Paleo/Data201803.xlsx','Modern Calibration');
%GDGT0 GDGT1 GDGT2 GDGT3 Cren Cren' long lat Temp Depth
cretaceous=xlsread('~/Work/Paleo/Data201803.xlsx','Cretaceous');
eocene=xlsread('~/Work/Paleo/Data201803.xlsx','Eocene');

eocenelat=xlsread('~/Work/Paleo/EoceneLatitude201808.xlsx');
cretaceouslat=xlsread('~/Work/Paleo/CretaceousLatitude201808.xlsx');
eocene=[eocene';eocenelat(2:end,5:6)']'; % add latitude and early/mid/late columns
cretaceous=[cretaceous';cretaceouslat(3:end,5)']';

eocenecut=eocene(~isnan(eocene(:,1)),:);


stdmodern=(std(modern));


clear distmin; clear distsq; clear temptrue; clear tempguess;
validation=load('~/Work/Paleo/validation.txt');
temptrue=modern(validation,9);
Nval=length(validation);
valcount=numel(validation)/length(validation);

for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    for(i=1:Nval),
        for(j=1:length(calibration)),
            dist=(modern(calibration(j),1:6)-modern(validation(i,k),1:6))./stdmodern(1:6);
            distsq(calibration(j))=sqrt(sum(dist.^2));
        end;
        distsq(validation(:,k))=inf; 
        [distmin(i,k),index(i,k)]=min(distsq);
    end;
end;

figure(1), set(gca, 'FontSize', 16);
hist(distmin(:),100); 
set(gca, 'FontSize', 24); 
%xlabel('Normalized distance to nearest calibration point')%, title('Modern')
xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex'), 

tempguess=modern(index,9);
figure(2), set(gca, 'FontSize', 16);
scatter(distmin(:),tempguess(:)-temptrue(:),'filled');
set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex'), 
ylabel('$\hat{T}_\mathrm{nearest} - T $', 'Interpreter', 'latex')%, title('Modern')
mean(abs(temptrue-tempguess))
mean(temptrue-tempguess)
std(temptrue-tempguess)
std(modern(:,9))

TEX86modern=(modern(:,3)+modern(:,4)+modern(:,6))./...
        (modern(:,2)+modern(:,3)+modern(:,4)+modern(:,6));
figure(3), scatter(TEX86modern,modern(:,9),'filled');
set(gca, 'FontSize', 20), xlabel('TEX_{86}'), ylabel('Temperature');
corr(TEX86modern,modern(:,9))


for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    for(i=1:Nval),
        for(j=1:length(calibration)),
            dist=(TEX86modern(calibration(j))-TEX86modern(validation(i,k)));
            distsq(calibration(j))=sqrt(sum(dist.^2));
        end;
        distsq(validation(:,k))=inf; 
        [distmin(i,k),index(i,k)]=min(distsq);
    end;
end;
tempguess=modern(index,9);
figure(4), set(gca, 'FontSize', 16);
scatter(distmin(:),tempguess(:)-temptrue(:),'filled');
set(gca, 'FontSize', 24); 
xlabel('Distance to nearest TEX$_{86}$ point',  'Interpreter', 'latex'), 
ylabel('$\hat{T}_\mathrm{nearest\ TEX} - T$', 'Interpreter', 'latex')%, title('Modern')
mean(abs(temptrue-tempguess))
mean(temptrue-tempguess)
std(temptrue-tempguess)
std(modern(:,9))

%OneTEX
for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    p=polyfit(1./TEX86modern(calibration),modern(calibration,9),1)
    tempguess(:,k) = polyval(p,1./TEX86modern(validation(:,k)));
end;
std(temptrue-tempguess(:))
std(modern(:,9))
OneTEX=54.5-19.1./TEX86modern(validation);
std(temptrue-OneTEX(:))

TEXH=38.6+68.4*log(TEX86modern)/log(10);
TEXL=67.5*modern(:,1)+46.9;
std(temptrue'-TEXH(1:floor(length(modern)/2)))
std(temptrue'-OneTEX(1:floor(length(modern)/2)))

%RegTree=fitensemble(modern(1:floor(length(modern)/2),1:6),modern(1:floor(length(modern)/2),9),'LSBoost',100,'Tree');
%RegTree=fitensemble(modern(floor(length(modern)/2)+1:length(modern),1:6),modern(floor(length(modern)/2)+1:length(modern),9),'LSBoost',100,'Tree');

clear tempguess;
%random forest
for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    RegTree=fitensemble(modern(calibration,1:6),modern(calibration,9),'LSBoost',100,'Tree');
    tempguess(:,k)=predict(RegTree,modern(validation(:,k),1:6));
end;
figure(7); set(gca, 'FontSize', 16); scatter(temptrue,tempguess(:)-temptrue,'filled'); 
set(gca, 'FontSize', 24); 
xlabel('$T$',  'Interpreter', 'latex'), 
ylabel('$T-\hat{T}_\mathrm{random\ forest}$', 'Interpreter', 'latex')
std(temptrue-tempguess(:))


clear tempguess; clear tempstd;
%GP regression
for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    %gprMdl = fitrgp(modern(calibration,1:6),modern(calibration,9));
    gprMdl = fitrgp(modern(calibration,1:6),modern(calibration,9),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',std(modern(:,[1:6,9])),'Sigma',std(modern(:,9)));
    [tempguess(:,k),tempstd(:,k)]=predict(gprMdl,modern(validation(:,k),1:6));
    L=loss(gprMdl,modern(validation(:,k),1:6),modern(validation(:,k),9));
    %gprMdl.KernelInformation.KernelParameters./(std(modern(:,[1:6,9])))'
    sqrt(L)
end;
figure(8); set(gca, 'FontSize', 16); scatter(temptrue,tempguess(:)-temptrue,'filled'); 
set(gca, 'FontSize', 24); 
xlabel('$T$',  'Interpreter', 'latex'), 
ylabel('$T-\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
std(temptrue-tempguess(:))

figure(9); set(gca, 'FontSize', 16); 
[pred,predstd,pred95]=predict(gprMdl,modern(validation(:,k),1:6),'Alpha',0.05);
%plot(modern(validation(:,k),9),[modern(validation(:,k),9)';pred';pred95(:,1)'; pred95(:,2)'],...
%    'LineWidth', 2);
plot(modern(validation(:,k),9),modern(validation(:,k),9)); hold on;
scatter(modern(validation(:,k),9),pred,'*');
scatter(modern(validation(:,k),9),pred95(:,1),'^');
scatter(modern(validation(:,k),9),pred95(:,2),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('$T$',  'Interpreter', 'latex'), 
ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('True T', 'GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'NorthWest');
sum(modern(validation(:,k),9)>=pred95(:,1) & modern(validation(:,k),9)<=pred95(:,2))./Nval

%p-p plot
interval=5:5:95;
for(i=1:length(interval))
    nsigma=sqrt(2)*erfinv(interval(i)/100);
    lowerbound=tempguess-nsigma*tempstd; 
    upperbound=tempguess+nsigma*tempstd;
    fracininterval(i)=sum(temptrue>=lowerbound(:) & temptrue<=upperbound(:))./(10*Nval);
end;
figure(10); set(gca, 'FontSize', 16); 
plot(interval/100,fracininterval,'LineWidth',3); hold on;
plot([0:100]/100, [0:100]/100,'k:'); hold off;
set(gca, 'FontSize', 24)
ylabel('Fraction of true T within interval'), xlabel('Confidence interval')




k=1;
arr=[1:length(modern)];
calibrationindex=~ismember(1:length(modern),validation(:,k));
calibration=arr(calibrationindex);
gprMdl = fitrgp(modern(calibration,1:6),modern(calibration,9),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',std(modern(:,[1:6,9])),'Sigma',std(modern(:,9)));
tempguess=predict(gprMdl,modern(validation(:,k),1:6));
L=loss(gprMdl,modern(validation(:,k),1:6),modern(validation(:,k),9));
sigmaL = gprMdl.KernelInformation.KernelParameters(1:end-1);
sqrt(L)
for(i=1:length(validation(:,k))),
    for(j=1:length(calibration)),
            dist=(modern(calibration(j),1:6)-modern(validation(i,k),1:6))./stdmodern(1:6);
            distsq(j)=sqrt(sum(dist.^2));
            distw=(modern(calibration(j),1:6)-modern(validation(i,k),1:6))./sigmaL';
            distsqw(j)=sqrt(sum(distw.^2));
    end;
    [distmin(i),index(i)]=min(distsq);
    [distwmin(i),indexw(i)]=min(distsqw);
end;
figure(11); set(gca, 'FontSize', 16); scatter(distmin,tempguess-modern(validation(:,k),9),'filled'); 
set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex'), 
ylabel('$T-\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
corr(distmin',abs(tempguess-modern(validation(:,k),9)))
figure(12); set(gca, 'FontSize', 16); scatter(distwmin,tempguess-modern(validation(:,k),9),'filled'); 
set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest,weighted}$','Interpreter', 'latex'), 
ylabel('$T-\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
corr(distwmin',abs(tempguess-modern(validation(:,k),9)))

clear tempguess;
%weighted nearest neigbours
for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    for(i=1:Nval),
        for(j=1:length(calibration)),
            dist=(TEX86modern(calibration(j))-TEX86modern(validation(i,k)));
            distsq(calibration(j))=sqrt(sum(dist.^2));
        end;
        distsq(validation(:,k))=inf; 
        weights=1./(0.0001+distsq).^2; %add 0.01 for regularisation
        sumweights=sum(weights);
        weights=weights/sumweights;
        tempguess(i,k)=weights*modern(:,9);
    end;
end;
figure(13); set(gca, 'FontSize', 16); scatter(temptrue,tempguess(:)-temptrue,'filled'); 
set(gca, 'FontSize', 24); 
xlabel('$T$',  'Interpreter', 'latex'), 
ylabel('$T-\hat{T}_\mathrm{weighted\ neighbours}$', 'Interpreter', 'latex')
std(temptrue-tempguess(:))


clear distmin; clear distsq;
for(i=1:length(eocene)),
    for(j=1:length(modern)),
        dist=(modern(j,1:6)-eocene(i,1:6))./stdmodern(1:6);
        distsq(j)=sqrt(sum(dist.^2));
    end;
    distmin(i)=min(distsq);
end;
figure(14), set(gca, 'FontSize', 16);
hist(distmin(:),100); set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest}$ for Eocene samples','Interpreter', 'latex')
quantile(distmin(:),0.33)

clear distmin; clear distsq;
for(i=1:length(cretaceous)),
    for(j=1:length(modern)),
        dist=(modern(j,1:6)-cretaceous(i,1:6))./stdmodern(1:6);
        distsq(j)=sqrt(sum(dist.^2));
    end;
    distmin(i)=min(distsq);
end;
figure(15), set(gca, 'FontSize', 16);
hist(distmin(:),100); set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest}$ for Cretaceous samples','Interpreter', 'latex')



%GP regression on full modern data set, applied to eocene and cretaceous
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

clear tempeocene, clear tempeocenestd, clear tempeocene95
clear tempcretaceous, clear tempcretaceousstd, clear tempcretaceous95
[tempeocene,tempeocenestd,tempeocene95]=predict(gprMdl,eocenecut(:,1:6),'Alpha',0.05);
[tempcretaceous,tempcretaceousstd,tempcretaceous95]=predict(gprMdl,cretaceous(:,1:6),'Alpha',0.05);
mean(tempmodern), mean(tempeocene), mean(tempcretaceous)

figure(16); set(gca, 'FontSize', 16); 
centers=[-6:3:30];
edges=[centers-1.5, max(centers)+1.5];
histmodern=histcounts(tempmodern,edges);
histeocene=histcounts(tempeocene,edges);
histcretaceous=histcounts(tempeocene,edges);
%bar(centers,histpred)
histogram(tempmodern,edges); hold on; histogram(tempeocene,edges); histogram(tempcretaceous,edges); hold off;
set(gca, 'FontSize', 24); 
xlabel('Predicted temperature'); ylabel('Counts');
legend('Modern','Eocene','Cretaceous');

figure(17); set(gca, 'FontSize', 16);
centers=[0:0.5:10];
edges=[centers-0.25, max(centers)+0.25];
histogram(tempmodernstd,edges); hold on; histogram(tempeocenestd,edges); histogram(tempcretaceousstd,edges); hold off;
set(gca, 'FontSize', 24); 
xlabel('Prediction standard deviation'); ylabel('Counts');
legend('Modern','Eocene','Cretaceous');


for(i=1:length(eocenecut)),
    for(j=1:length(modern)),
            dist=(modern(j,1:6)-eocenecut(i,1:6))./sigmaL';
            distsq(j)=sqrt(sum(dist.^2));
    end;
    [distmineocene(i),indexeocene(i)]=min(distsq);
end;

for(i=1:length(cretaceous)),
    for(j=1:length(modern)),
            dist=(modern(j,1:6)-cretaceous(i,1:6))./sigmaL';
            distsq(j)=sqrt(sum(dist.^2));
    end;
    [distmincretaceous(i),indexcretaceous(i)]=min(distsq);
end;

figure(18), set(gca, 'FontSize', 16);
%hist(distmin(:),100); 
scatter(log(distmineocene)/log(10),tempeocenestd, 'filled'); hold on;
plot(log(0.5)/log(10)*ones(size([3:9])),[3:9],'k:'); hold off;
set(gca, 'FontSize', 24); 
%xlabel('Normalized distance to nearest calibration point')%, title('Modern')
xlabel('$\log_{10}(D_\mathrm{nearest,weighted})$','Interpreter', 'latex'),
ylabel('Eocene prediction standard deviation')

figure(19), set(gca, 'FontSize', 16);
%hist(distmin(:),100); 
scatter(log(distmincretaceous)/log(10),tempcretaceousstd, 'filled'); hold on;
plot(log(0.5)/log(10)*ones(size([3:9])),[3:9],'k:'); hold off;
set(gca, 'FontSize', 24); 
%xlabel('Normalized distance to nearest calibration point')%, title('Modern')
xlabel('$\log_{10}(D_\mathrm{nearest,weighted})$','Interpreter', 'latex'),
ylabel('Cretaceous prediction standard deviation')


figure(17); set(gca, 'FontSize', 16);
centers=[0:0.5:10];
edges=[centers-0.25, max(centers)+0.25];
histogram(tempmodernstd,edges); hold on; histogram(tempeocenestd,edges); histogram(tempcretaceousstd,edges); 
histogram(tempeocenestd(log(distmineocene)/log(10)<log(0.5)/log(10)),edges); 
histogram(tempcretaceousstd(log(distmincretaceous)/log(10)<log(0.5)/log(10)),edges); hold off;
set(gca, 'FontSize', 24); 
xlabel('Prediction standard deviation'); ylabel('Counts');
legend('Modern','Eocene','Cretaceous','Eocene after cut', 'Cretaceous after cut');


%%%% Latitudes
figure(20);
set(gca, 'FontSize', 16); 
scatter(modern(:,8),modern(:,9),'filled'); hold on;
scatter(modern(:,8),tempmodern,'*');
scatter(modern(:,8),tempmodern95(:,1),'^');
scatter(modern(:,8),tempmodern95(:,2),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Latitude [degrees]'), 
ylabel('Temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('True T', 'GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'South');
sum(modern(:,9)>=tempmodern95(:,1) & modern(:,9)<=tempmodern95(:,2))./length(modern)


figure(21);
set(gca, 'FontSize', 16); 
eoceneselect=log(distmineocene)/log(10)<log(0.5)/log(10);
scatter(eocenecut(eoceneselect,7),tempeocene(eoceneselect),'*'); hold on;
scatter(eocenecut(eoceneselect,7),tempeocene95(eoceneselect,1),'^');
scatter(eocenecut(eoceneselect,7),tempeocene95(eoceneselect,2),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Eocene latitude [degrees]'), 
ylabel('Eocene temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'South');

figure(22);
set(gca, 'FontSize', 16); 
eoceneselect=log(distmineocene)/log(10)<log(0.5)/log(10);
scatter(eocene(eoceneselect'&eocenecut(:,8)==1,7),...
    tempeocene(eoceneselect'&eocenecut(:,8)==1),'filled'); hold on;
scatter(eocene(eoceneselect'&eocenecut(:,8)==2,7),...
    tempeocene(eoceneselect'&eocenecut(:,8)==2),'filled');
scatter(eocene(eoceneselect'&eocenecut(:,8)==3,7),...
    tempeocene(eoceneselect'&eocenecut(:,8)==3),'filled');
scatter(eocene(~eoceneselect,7),tempeocene(~eoceneselect),20,[0.5 0.5 0.5], 'filled'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Eocene latitude [degrees]'), 
ylabel('Eocene temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('Early; Trusted GP regression predictor', 'Middle; Trusted GP regression predictor',...
    'Late; Trusted GP regression predictor', 'Untrusted GP regression predictor',...
    'Location', 'South');
 
scatter(cretaceous(cretaceousselect,7),tempcretaceous(cretaceousselect),'filled'); hold on;
scatter(cretaceous(~cretaceousselect,7),tempcretaceous(~cretaceousselect),20,[0.5 0.5 0.5], 'filled'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Cretaceous latitude [degrees]'), 
ylabel('Cretaceous temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('Trusted GP regression predictor', 'Untrusted GP regression predictor',...
    'Location', 'South');


figure(23);
set(gca, 'FontSize', 16); 
cretaceousselect=log(distmincretaceous)/log(10)<log(0.5)/log(10);
scatter(cretaceous(cretaceousselect,7),tempcretaceous(cretaceousselect),'*'); hold on;
scatter(cretaceous(cretaceousselect,7),tempcretaceous95(cretaceousselect,1),'^');
scatter(cretaceous(cretaceousselect,7),tempcretaceous95(cretaceousselect,2),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Cretaceous latitude [degrees]'), 
ylabel('Cretaceous temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'South');

figure(24);
set(gca, 'FontSize', 16); 
cretaceousselect=log(distmincretaceous)/log(10)<log(0.5)/log(10);
%s=scatter(cretaceous(~cretaceousselect,7),tempcretaceous(~cretaceousselect),'filled'); 
%s.MarkerEdgeColor='w'; s.MarkerFaceColor=[0.5 0.5 0.5]; 
scatter(cretaceous(cretaceousselect,7),tempcretaceous(cretaceousselect),'filled'); hold on;
scatter(cretaceous(~cretaceousselect,7),tempcretaceous(~cretaceousselect),20,[0.5 0.5 0.5], 'filled'); hold off;
set(gca, 'FontSize', 24); 
xlabel('Cretaceous latitude [degrees]'), 
ylabel('Cretaceous temperature'),
%ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('Trusted GP regression predictor', 'Untrusted GP regression predictor',...
    'Location', 'South');

outcretaceous=[tempcretaceous';tempcretaceousstd';distmincretaceous;cretaceousselect]';
outeocene=[tempeocene';tempeocenestd';distmineocene;eoceneselect]';
outeoceneuncut=zeros(length(eocene),4);
outeoceneuncut(~isnan(eocene(:,1)),:)=outeocene;
outeoceneuncut(isnan(eocene(:,1)),:)=nan;
csvwrite('EocenePredictions.csv',outeoceneuncut);
csvwrite('CretaceousPredictions.csv',outcretaceous);

%no lat/long on cretaceous/eocene?

%N=length(modern);
%K=floor(N/10);
%X=[];
%for(i=1:10),
%    X=[X; randperm(N,K)];
%end;
%save('~/Work/Paleo/validation.txt','X','-ascii','-tabs');