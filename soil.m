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

tempguess=zeros(K,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input(calibration,:),temp(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(temp)],'Sigma',std(temp));
    tempguess(:,k)=predict(gprMdl,input(validation(:,k),:));
    L=loss(gprMdl,input(validation(:,k),:),temp(validation(:,k)));
    sqrt(L)
end;
std(tempguess-temp(validation))
mean(std(tempguess-temp(validation)))
std(temp)
Rsqtemp=1-sum(sum((tempguess-temp(validation)).^2))/sum(sum((temp(validation)-mean(temp)).^2))


pHguess=zeros(K,10);
for(k=1:10),
    arr=[1:length(input)];
    calibrationindex=~ismember(1:length(input),validation(:,k));
    calibration=arr(calibrationindex);
    gprMdl = fitrgp(input(calibration,:),pH(calibration),...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',[std(input) std(pH)],'Sigma',std(pH));
    pHguess(:,k)=predict(gprMdl,input(validation(:,k),:));
    L=loss(gprMdl,input(validation(:,k),:),pH(validation(:,k)));
    sqrt(L)
end;
std(pHguess-pH(validation))
mean(std(pHguess-pH(validation)))
std(pH)
RsqpH=1-sum(sum((pHguess-pH(validation)).^2))/sum(sum((pH(validation)-mean(pH)).^2))
