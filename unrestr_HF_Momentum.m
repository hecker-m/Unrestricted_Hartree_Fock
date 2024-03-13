close all;clear all;
global count Nx Ny Nz s0 sx sy sz dpar dz u v w Temp eps SigmaX SigmaZ penalty mu Nb
s0=[1 0;0 1];
sx=[0 1;1 0];
sy=[0 -1i;1i 0];
sz=[1 0;0 -1];
penalty=0;
dpar=1;
dz=0.3;
u=1;
v=0.3*u;
w=0.08*u;
Temp=100.0;

eps=10^(-8);
SigmaX=kron(sx,diag(ones(1,2)));
SigmaZ=kron(sz,diag(ones(1,2)));
count=0;

%We choose a particle density of 2 per site.
Nb=2;

%size of the kx,ky,kz grid
Nx=1+2^3;
Ny=1+2^3;
Nz=1+2^3;



%
% A . x <= b 
%simply ensure that R>0 throughout the routine
%By hand, we also choose the Ising variable CB1g >0.
A=zeros(8,8);
A(1,1)=-1;
A(2,2)=-1;
b=zeros(8,1);


initial = [1.1, 80.1, 0.05, 0.083, 0.12, 0.012, 0.007, 0.0012];    % Number of variables
[x,fval]=fmincon(@free_energy,initial,A,b,[],[],[],[],@mycon);


%Two imposed constraints, see below.
function [c, ceq]=mycon(x)
    c=constr_primary_order(x);
    ceq=constr_free_energy(x);
end

%As the implemented theory is only applicable above Tc
%we have to ensure that y<0 throughout
function y=constr_primary_order(x)
    global mu Nb u v w
    nA1g=Nb;%x(2)=Nb-x(1)
    nB1g=2*x(1)-Nb;
    nB2g=2*x(2);
    nA2g=2*x(3);
    xA1g=x(4)+(x(5)+1i*x(6));
    xB1g=x(4)-(x(5)+1i*x(6));
    xB2g=2*(x(7)+1i*x(8));
    R=-(mu+(3*u+v+w)/6)+(3*u+v+w)/12*nA1g;
    CE1=(u-v+3*w)/12*nB1g;
    CE2=(u-v-w)/12*nB2g;
    CA2=(u+3*v-w)/12*nA2g;
    cA1=(u-v+w)/12*xA1g;
    cE1=(u+v+w)/12*xB1g;
    cE2=(u+v-w)/12*xB2g;
    y=-(R^2-CE1^2-CE2^2-CA2^2-abs(cA1)^2-abs(cE1)^2-abs(cE2)^2);
end

%We are making sure that the imaginary parts of the eigenvalues
%remain zero throughout the minimization
function y=constr_free_energy(x)
    global Nx Ny Nz SigmaZ mu Nb
    nA1g=Nb;%x(2)=Nb-x(1)
    nB1g=2*x(1)-Nb;
    nB2g=2*x(2);
    nA2g=2*x(3);
    xA1g=x(4)+(x(5)+1i*x(6));
    xB1g=x(4)-(x(5)+1i*x(6));
    xB2g=2*(x(7)+1i*x(8));

    y=0;
    for nx=-(Nx-1)/2:(Nx-1)/2
        kx=2*pi*nx/Nx;
        for ny=-(Ny-1)/2:(Ny-1)/2
            ky=2*pi*ny/Ny;
            for nz=-(Nz-1)/2:(Nz-1)/2
                kz=2*pi*nz/Nz;
                HBdG=HBdG_fcn(kx,ky,kz,mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
                HBdGtilde=SigmaZ*HBdG;
                evs=eig(HBdGtilde);
                [~,ind]=sort(real(evs),'descend');
                evs=evs(ind);
                y=y+abs(imag(evs(1,1)))+abs(imag(evs(2,1)));
            end
        end
    end
end

%Constructs the matrix T associated with the eigenvector matrix Us.
%Note that eigenvectors associated with a degenerate subspace
%have to be re-orthogonalized, see notes for the correct application
%of SigmaX and SigmaZ.
function Tmat=T_constr(Us,evs,N)
    global eps SigmaX SigmaZ
    Tmat=zeros(2*N,2*N);
    ell=1;
    Tmat(:,1)=Us(:,1)/sqrt(abs(ctranspose(Us(:,1))*SigmaZ*Us(:,1)));
    for i=2:N
        if abs(evs(i,1)-evs(i-1,1))<eps
            ell=ell+1;
            normadd=0;
            for ellp=1:ell-1
                normadd=normadd+abs(ctranspose(Tmat(:,i-ellp))*SigmaZ*Us(:,i))^2;
            end
            numadd=0*Us(:,i);
            for ellp=1:ell-1
                numadd=numadd+(ctranspose(Tmat(:,i-ellp))*SigmaZ*Us(:,i))*Tmat(:,i-ellp);
            end
            Tmat(:,i)=(Us(:,i)-numadd)/sqrt(abs(ctranspose(Us(:,i))*SigmaZ*Us(:,i)-normadd));
        else
            ell=1;
            Tmat(:,i)=Us(:,i)/sqrt(abs(ctranspose(Us(:,i))*SigmaZ*Us(:,i)));
        end
    end
    for i=1:N
        Tmat(:,N+i)=SigmaX*conj(Tmat(:,i));
    end
end

%Computes the current value of the particle number average
%Note that this involves the eigenvalue matrix T,
%for which the correct properties have to be imposed, see notes.
function [Nb_approx,test]=Nb_fcn(mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g)
    global Nx Ny Nz eps SigmaZ
    Nb_approx=0;test=0;
    for nx=-(Nx-1)/2:(Nx-1)/2
        kx=2*pi*nx/Nx;
        for ny=-(Ny-1)/2:(Ny-1)/2
            ky=2*pi*ny/Ny;
            for nz=-(Nz-1)/2:(Nz-1)/2
                kz=2*pi*nz/Nz;
                HBdG=HBdG_fcn(kx,ky,kz,mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
                HBdGtilde=SigmaZ*HBdG;
                [Uev,D]=eig(HBdGtilde);
                [~,ind]=sort(real(diag(D)),'descend');
                Ds=D(ind,ind);
                evs=diag(Ds);
                Us=Uev(:,ind);
                testStep=test_fcn(evs,eps);
                if testStep==1
                    test=1;
                    break
                end
                TMat=T_constr(Us,evs,2);
                U=TMat(1:2,1:2);
                V=TMat(1:2,3:4);
                nbmat=nb_fcn(evs,2);
                Nb_approx=Nb_approx+trace(transpose(V)*conj(V)+(transpose(V)*conj(V)+transpose(U)*conj(U))*nbmat);
            end
            if test==1
                break
            end
        end
        if test==1
            break
        end
    end
    Nb_approx=real(Nb_approx)/Nx/Ny/Nz;
end

%As we keep the particle number Nb fixed, 
%we have to always adjust the chemical potential
%Finds the current chemical potential through interval nesting
function mu_approx=mu_det(nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g)
    global Nb u v w
    mu1=-0.001;
    [~,test1]=Nb_fcn(mu1,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
    %Moves the value close to its actual value
    while test1==1
        mu1=mu1-0.1;
        [~,test1]=Nb_fcn(mu1,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
        %disp(mu1);
    end
    %interval nesting routine
    mu2=min([mu1-12,10*mu1]);mu3=(mu1+mu2)/2;
    [Nb_approx3,~]=Nb_fcn(mu3,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
    while abs(Nb_approx3-Nb)>0.00001
        mu3=(mu1+mu2)/2;
        [Nb_approx1,test1]=Nb_fcn(mu1,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
        [Nb_approx2,test2]=Nb_fcn(mu2,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
        [Nb_approx3,~]=Nb_fcn(mu3,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
        if Nb_approx2>Nb
            %disp('mu-guess too low');
            mu2=2*mu2;
        elseif Nb_approx1<Nb
            R=-(mu1+(3*u+v+w)/6)+(3*u+v+w)/12*nA1g;
            CE1=(u-v+3*w)/12*nB1g;
            CE2=(u-v-w)/12*nB2g;
            CA2=(u+3*v-w)/12*nA2g;
            cA1=(u-v+w)/12*xA1g;
            cE1=(u+v+w)/12*xB1g;
            cE2=(u+v-w)/12*xB2g;
            out=['mu not satisfiable: mu1=',num2str(mu1),' ,R1=', num2str(R),' ,CB1g=', num2str(drop(CE1)),
                ' ,CB2g=', num2str(drop(CE2)),...
                ' ,CA2g=', num2str(drop(CA2)),' ,cA1g=', num2str(drop(cA1)),' ,cB1g=', num2str(drop(cE1)),' ,cB2g=', num2str(drop(cE2))];             
            disp(out);
            break;
        elseif Nb_approx3-Nb>0
            mu1=mu3;
        elseif Nb_approx3-Nb<0
            mu2=mu3;   
        end
    end
    mu_approx=mu3;
end

%Computes the free energy for the given configuration x
function y=free_energy(x)
    global count Nx Ny Nz u v w Temp SigmaZ mu Nb
    nA1g=Nb;%x(2)=Nb-x(1)
    nB1g=2*x(1)-Nb;
    nB2g=2*x(2);
    nA2g=2*x(3);
    xA1g=x(4)+(x(5)+1i*x(6));
    xB1g=x(4)-(x(5)+1i*x(6));
    xB2g=2*(x(7)+1i*x(8));
    %Computes the current chemical potential mu
    mu=mu_det(nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
    
    y=Nx*Ny*Nz*(mu+(3*u+v+w)/6)+Nx*Ny*Nz*(mu-(3*u+v+w)/12)*nA1g;
    for nx=-(Nx-1)/2:(Nx-1)/2
        kx=2*pi*nx/Nx;
        for ny=-(Ny-1)/2:(Ny-1)/2
            ky=2*pi*ny/Ny;
            for nz=-(Nz-1)/2:(Nz-1)/2
                kz=2*pi*nz/Nz;
                HBdG=HBdG_fcn(kx,ky,kz,mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
                HBdGtilde=SigmaZ*HBdG;
                evs=eig(HBdGtilde);
                [~,ind]=sort(real(evs),'descend');
                evs=evs(ind);
                y=y-(1/2*evs(1,1)+Temp*log(nb_scal_fcn(evs(1,1))))-(1/2*evs(2,1)+Temp*log(nb_scal_fcn(evs(2,1))));
            end
        end
    end
    count=count+1;
    %printing the current values, after being rewritten into the composite channels.
    if mod(count,1)==0
        R=-(mu+(3*u+v+w)/6)+(3*u+v+w)/12*nA1g;
        CE1=(u-v+3*w)/12*nB1g;
        CE2=(u-v-w)/12*nB2g;
        CA2=(u+3*v-w)/12*nA2g;
        cA1=(u-v+w)/12*xA1g;
        cE1=(u+v+w)/12*xB1g;
        cE2=(u+v-w)/12*xB2g;
        crit=R^2-CE1^2-CE2^2-CA2^2-abs(cA1)^2-abs(cE1)^2-abs(cE2)^2;

        [Nb_val,~]=Nb_fcn(mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g);
            out=['F=',num2str(y/Nx/Ny/Nz),' ,Crit=',num2str(crit),' ,mu=',num2str(mu),' ,R=', num2str(drop(R)),' ,CB1g=', num2str(drop(CE1)),' ,CB2g=', num2str(drop(CE2)),...
            ' ,CA2g=', num2str(drop(CA2)),' ,cA1g=', num2str(drop(cA1)),' ,cB1g=', num2str(drop(cE1)),' ,cB2g=', num2str(drop(cE2)),' ,Nb=', num2str(drop(Nb_val))]; 
        disp(out);
    end
    y=real(y)/Nx/Ny/Nz;
end


%Constructs the 4x4 BdG-Matrix
function HBdG=HBdG_fcn(kx,ky,kz,mu,nA1g,nA2g,nB1g,nB2g,xA1g,xB1g,xB2g)
    global s0 sx sy sz u v w
    muTilde=mu+(3*u+v+w)/6;
    h00=(fA1g(kx,ky,kz)-muTilde+(3*u+v+w)/12*(nA1g))*s0...
        +(u-v+3*w)/12*(nB1g)*sz...
        +(u-v-w)/12*(nB2g)*sx...
        +(u+3*v-w)/12*(nA2g)*sy;
    x00=(u-v+w)/12*(xA1g)*s0...
        +(u+v+w)/12*(xB1g)*sz...
        +(u+v-w)/12*(xB2g)*sx;
    HBdG=kron((s0+sz)/2,h00)+kron((s0-sz)/2,transpose(h00))+kron((sx+1i*sy)/2,x00)+kron((sx-1i*sy)/2,ctranspose(x00));
end

%checks whether the eigenvalues have non-zero imaginary parts
function test=test_fcn(evs,eps)
    test=0;     
    for i=1:4
        if abs(imag(evs(i,1)))>eps
            test=1;  
            break
        end
    end
end

%Bose function
function nb=nb_scal_fcn(omega)
    global Temp
    nb=1/(exp(omega/Temp)-1);
end

%Bose function
function nbmat=nb_fcn(evs,N)
    global Temp
    nbmat=zeros(N,N);
    for i=1:N
        nbmat(i,i)=1/(exp(evs(i,1)/Temp)-1);
    end
end

%A1g dispersion relation
function val=fA1g(kx,ky,kz)
    global dpar dz
    val=dpar*(2-cos(kx)-cos(ky))+dz*(1-cos(kz));
end

function val=drop(quant)
    if abs(quant)<10^(-5)
        val=0;
    else
        val=quant;
    end
end

