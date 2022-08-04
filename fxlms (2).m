clear all;
%% code for the fxlms algorithm
fre=50;
phi=pi/3;

samp_fre=20*fre;% sampling frequency as much as higher
dt=1/samp_fre; 
t=0:dt:10;
m=length(t);

%secondary path impulse response
A=2;
zeta=0.05;
wn=20;
wd=sqrt(1-zeta^2);
y_sec=A*(exp(-zeta*wn*t).*sin(wd*t-phi))'; 

x_2=zeros(m,1);
x_3=zeros(m,1);
 
% noise signal
x_r=sin(t*3+pi);

%reference signal
xr=sin(2*pi*fre*t);

% for weight
N=50;
ww=zeros(N,1); % for storing the weight
x_1=zeros(N,1);
y_fi=zeros(N,1);

%for error and output
err=zeros(m,1);
y_t=zeros(m,1);
mu=0.0001;

for n=1:m
    x_1=[x_r(n);x_1(1:N-1)];  %conv b/w weight and input
    y=sum(x_1.*ww);           
   
    x_2=[x_r(n); x_2(1:m-1)]; %conv b/w sec and input
    y_sout=sum(x_2.*y_sec);
    
    x_3=[y; x_3(1:m-1)];      %conv b/w y and sec
    out=sum(x_3.*y_sec);

    err(n)=xr(n)+out;         %finding the error

    y_fi=[y_sout;y_fi(1:N-1)]; 

    ww=ww-mu*err(n)*y_fi;     %updating the weight

    y_t(n)=out;               

end
plot(err)
