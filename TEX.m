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


stdmodern=(std(modern));


validation=load('~/Work/Paleo/validation.txt');
clear distmin; clear distsq;
Nval=length(validation);
valcount=numel(validation)/length(validation);
for(k=1:valcount),
    arr=[1:length(modern)];
    calibrationindex=~ismember(1:length(modern),validation(:,k));
    calibration=arr(calibrationindex);
    for(i=1:Nval),
        for(j=1:length(calibration)),
            dist=(modern(j,1:6)-modern(validation(i,k),1:6))./stdmodern(1:6);
            distsq(j)=sqrt(sum(dist.^2));
            if(find(j==validation(:,k))), distsq(j)=inf; end;
        end;
        [distmin(i,k),index(i,k)]=min(distsq);
    end;
end;

figure(1), set(gca, 'FontSize', 16);
hist(distmin(:),100); 
set(gca, 'FontSize', 24); 
%xlabel('Normalized distance to nearest calibration point')%, title('Modern')
xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex'), 

temptrue=modern(validation,9);
tempguess=modern(index,9);
figure(2), set(gca, 'FontSize', 16);
scatter(distmin(:),tempguess(:)-temptrue(:),'filled');
set(gca, 'FontSize', 24); 
xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex'), 
ylabel('$T - \hat{T}_\mathrm{nearest}$', 'Interpreter', 'latex')%, title('Modern')
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
    for(i=1:Nval),
        for(j=1:length(modern)),
            dist=(TEX86modern(j)-TEX86modern(validation(i,k)));
            distsq(j)=sqrt(sum(dist.^2));
            if(find(j==validation(:,k))), distsq(j)=inf; end;
        end;
        [distmin(i,k),index(i,k)]=min(distsq);
    end;
end;
tempguess=modern(index,9);
figure(4), set(gca, 'FontSize', 16);
scatter(distmin(:),tempguess(:)-temptrue(:),'filled');
set(gca, 'FontSize', 24); 
xlabel('Distance to nearest TEX$_{86}$ point',  'Interpreter', 'latex'), 
ylabel('$T - \hat{T}_\mathrm{nearest TEX}$', 'Interpreter', 'latex')%, title('Modern')
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
figure(6), set(gca, 'FontSize', 16);
scatter(EuFe(:,1),EuFe(:,2),'filled'); hold on;
plot(EuFe(:,1),yfit,'LineWidth',3); hold off;
set(gca, 'FontSize', 24);
xlabel('[Fe/H]'); ylabel('[Eu/H]'); title('Linear fit to data in EMP-RGB file');
legend('Data', 'Linear fit','Location','Best');

TEXH=38.6+68.4*log(TEX86modern)/log(10);
OneTEX=54.5-19.1./TEX86modern;
TEXL=67.5*modern(:,1)+46.9;
mean(temptrue'-TEXH(1:floor(length(modern)/2)))
mean(abs(temptrue'-TEXH(1:floor(length(modern)/2))))
std(temptrue'-TEXH(1:floor(length(modern)/2)))
mean(temptrue'-OneTEX(1:floor(length(modern)/2)))
mean(abs(temptrue'-OneTEX(1:floor(length(modern)/2))))
std(temptrue'-OneTEX(1:floor(length(modern)/2)))

RegTree=fitensemble(modern(1:floor(length(modern)/2),1:6),modern(1:floor(length(modern)/2),9),'LSBoost',100,'Tree');
RegTree=fitensemble(modern(floor(length(modern)/2)+1:length(modern),1:6),modern(floor(length(modern)/2)+1:length(modern),9),'LSBoost',100,'Tree');
prediction=predict(RegTree,modern(1:floor(length(modern)/2),1:6)); 
figure(7); scatter(temptrue,prediction,'filled'); set(gca, 'FontSize', 20);
mean(temptrue'-OneTEX(1:floor(length(modern)/2)))
mean(abs(temptrue'-OneTEX(1:floor(length(modern)/2))))
std(temptrue'-OneTEX(1:floor(length(modern)/2)))
mean(temptrue'-prediction)
mean(abs(temptrue'-prediction))
std(temptrue'-prediction)
mean((temptrue'-TEXH(1:floor(length(modern)/2))).^2)



for(i=1:length(eocene)),
    for(j=1:length(modern)),
        dist=(modern(j,1:6)-eocene(i,:))./stdmodern(1:6);
        distsq(j)=sqrt(sum(dist.^2));
    end;
    distmin(i)=min(distsq);
end;
figure(5)
hist(distmin,100); set(gca, 'FontSize', 20); 
xlabel('Normalized distance to closest calibration point'), title('Eocene')

for(i=1:length(cretaceous)),
    for(j=1:length(modern)),
        dist=(modern(j,1:6)-cretaceous(i,:))./stdmodern(1:6);
        distsq(j)=sqrt(sum(dist.^2));
    end;
    distmin(i)=min(distsq);
end;
figure(6)
hist(distmin,100); set(gca, 'FontSize', 20); 
xlabel('Normalized distance to closest calibration point'), title('Cretaceous')

%no lat/long on cretaceous/eocene?

%N=length(modern);
%K=floor(N/10);
%X=[];
%for(i=1:10),
%    X=[X; randperm(N,K)];
%end;
%save('~/Work/Paleo/validation.txt','X','-ascii','-tabs');