data=xlsread('~/Work/Paleo/SoilData201805.xlsx');
input=data(:,11:31);
pH=data(:,41);
temp=data(:,42);
precip=data(:,43);


gprMdlpH = fitrgp(input,pH,...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(pH)],'Sigma',std(pH));
%gprMdl.KernelInformation.KernelParameters./(std(modern(:,[1:6,9])))'    
[predpH,predpHstd,predpH95]=predict(gprMdlpH,input);
L=loss(gprMdlpH,input,pH);
sqrt(L)
std(predpH-pH)
std(pH)

gprMdltemp = fitrgp(input,temp,...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(temp)],'Sigma',std(temp));
%gprMdl.KernelInformation.KernelParameters./(std(modern(:,[1:6,9])))'    
[predtemp,predtempstd,predtemp95]=predict(gprMdltemp,input);
L=loss(gprMdltemp,input,temp);
sqrt(L)
std(predtemp-temp)
std(temp)

N=length(data);
K=floor(N/10);
validation=zeros(K,10);
for(i=1:10),
    validation(:,i)=(randperm(N,K))';
end;

temppred=zeros(K,10);
temppredstd=zeros(K,10);
temppred95=zeros(K,2,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input(calibration,:),temp(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(temp)],'Sigma',std(temp));
    log(gprMdl.KernelInformation.KernelParameters./([std(input) std(temp)])')/log(10)
    [temppred(:,k),temppredstd(:,k),temppred95(:,:,k)]=predict(gprMdl,input(validation(:,k),:));
    L=loss(gprMdl,input(validation(:,k),:),temp(validation(:,k)));
    sqrt(L)
end;
std(temppred-temp(validation))
mean(std(temppred-temp(validation)))
std(temp)
Rsqtemp=1-sum(sum((temppred-temp(validation)).^2))/sum(sum((temp(validation)-mean(temp)).^2))


pHpred=zeros(K,10);
pHpredstd=zeros(K,10);
pHpred95=zeros(K,2,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input(calibration,:),pH(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(pH)],'Sigma',std(pH));
    [pHpred(:,k),pHpredstd(:,k),pHpred95(:,:,k)]=predict(gprMdl,input(validation(:,k),:));
    L=loss(gprMdl,input(validation(:,k),:),pH(validation(:,k)));
    sqrt(L)
end;
std(pHpred-pH(validation))
mean(std(pHpred-pH(validation)))
std(pH)
RsqpH=1-sum(sum((pHpred-pH(validation)).^2))/sum(sum((pH(validation)-mean(pH)).^2))

figure(61);
plot(pH,pH); hold on;
pHval=pH(validation(:,:)); pHpredlow95=pHpred95(:,1,:); pHpredhigh95=pHpred95(:,2,:);
scatter(pHval(:),pHpred(:),'*');
scatter(pHval(:),pHpredlow95(:),'^');
scatter(pHval(:),pHpredhigh95(:),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('$pH$',  'Interpreter', 'latex'), 
ylabel('$\hat{pH}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('True pH', 'GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'NorthWest');
sum(pHval(:)>=pHpredlow95(:) & pHval(:)<=pHpredhigh95(:))./length(pHval(:))


figure(62);
plot(temp,temp); hold on;
tempval=temp(validation(:,:)); temppredlow95=temppred95(:,1,:); temppredhigh95=temppred95(:,2,:);
scatter(tempval(:),temppred(:),'*');
scatter(tempval(:),temppredlow95(:),'^');
scatter(tempval(:),temppredhigh95(:),'v'); hold off;
set(gca, 'FontSize', 24); 
xlabel('$T$',  'Interpreter', 'latex'), 
ylabel('$\hat{T}_\mathrm{GP\ regression}$', 'Interpreter', 'latex')
legend('True Temp', 'GP regression predictor', 'Lower 95% limit', 'Upper 95% limit',...
    'Location', 'NorthWest');
sum(tempval(:)>=temppredlow95(:) & tempval(:)<=temppredhigh95(:))./length(tempval(:))

input1517=data(:,[22,23,24,27,28,29]);
input1517comb=data(:,[35,36]);

temppred1517=zeros(K,10);
temppredstd1517=zeros(K,10);
temppred951517=zeros(K,2,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input1517(calibration,:),temp(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input1517) std(temp)],'Sigma',std(temp));
    [temppred1517(:,k),temppredstd1517(:,k),temppred951517(:,:,k)]=predict(gprMdl,input1517(validation(:,k),:));
    L=loss(gprMdl,input1517(validation(:,k),:),temp(validation(:,k)));
    sqrt(L)
end;
std(temppred1517-temp(validation))
mean(std(temppred1517-temp(validation)))
std(temppred1517(:)-temp(validation(:)))
std(temp)
Rsqtemp1517=1-sum(sum((temppred1517-temp(validation)).^2))/sum(sum((temp(validation)-mean(temp)).^2))

temppredlow951517=temppred951517(:,1,:); temppredhigh951517=temppred951517(:,2,:);
sum(tempval(:)>=temppredlow951517(:) & tempval(:)<=temppredhigh951517(:))./length(tempval(:))

pHpred1517=zeros(K,10);
pHpredstd1517=zeros(K,10);
pHpred951517=zeros(K,2,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input1517(calibration,:),pH(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input1517) std(pH)],'Sigma',std(pH));
    [pHpred1517(:,k),pHpredstd1517(:,k),pHpred951517(:,:,k)]=predict(gprMdl,input1517(validation(:,k),:));
    L=loss(gprMdl,input1517(validation(:,k),:),pH(validation(:,k)));
    sqrt(L)
end;
std(pHpred1517-pH(validation))
mean(std(pHpred1517(:)-pH(validation(:))))
std(pH)
sqrt(mean((pHpred1517-pH(validation)).^2))
RsqpH1517=1-sum(sum((pHpred1517-pH(validation)).^2))/sum(sum((pH(validation)-mean(pH)).^2))

temppred1517comb=zeros(K,10);
temppredstd1517comb=zeros(K,10);
temppred951517comb=zeros(K,2,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input1517comb(calibration,:),temp(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input1517comb) std(temp)],'Sigma',std(temp));
    [temppred1517comb(:,k),temppredstd1517comb(:,k),temppred951517comb(:,:,k)]...
        =predict(gprMdl,input1517comb(validation(:,k),:));
    L=loss(gprMdl,input1517comb(validation(:,k),:),temp(validation(:,k)));
    sqrt(L)
end;
std(temppred1517comb-temp(validation))
mean(std(temppred1517comb-temp(validation)))
std(temp)
Rsqtemp1517comb=1-sum(sum((temppred1517comb-temp(validation)).^2))/sum(sum((temp(validation)-mean(temp)).^2))
