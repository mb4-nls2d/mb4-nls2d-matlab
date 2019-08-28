%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  comparison of several numerical methods for 2D-NLS equation
%%  Domain: [0,2pi]^2
%%  boundary condition: periodic
%%  choice of numerical integrators
%%  + 4-stage 4th order explicit Runge kutta method
%%  + 2nd and 4th order Gauss mryhofd (which are symplectic Runge-Kutta methods)
%%  + 2nd order energy preserving (AVF) method (Quispel-McLaren 2008)
%%  + 4th order energy preserving methods
%%    (AVF collocation (Hairer 2010), parellel (Miyatake-Butcher 2016))
%%
%%  equation : iu_t = A u + eps |u|^2 u
%%                    A = -(d2/dx2 + d2/dy2)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function main_solution_2d()
solvers = ['RK4   '; 'GAUSS2'; 'GAUSS4'; 'AVF2  '; 'AVF4  '; 'MB4   '];
n = 7;
times = zeros(6, n);
for s = 1 : 6
    solvers(s, :)
    for scale = 1 : n
        scale
        times(s, scale) = main_solution_2d_aux(solvers(s, :), scale);
    end
end
for s = 1 : 6
    fprintf('%s ', solvers(s, :));
    fprintf('%.6f ', times(s, :));
    fprintf('\n');
end
end

function time = main_solution_2d_aux(type, scale)
%clear all
%close all
ode45tol=3e-14;% tolerance for ode45
TolX=1e-13; % torelance for simplified Newton method
quadtol =1e-12;

%% parameter
eps=0.1;

%% space mesh size
%scale = 8;
L1 = 2*pi * scale; % domain = [0,L1]*[0,L2]
L2 = 2*pi * scale;
M1 = 10 * scale; % # of grid points (x-axis)
M2 = 10 * scale; % # of grid points (y-axis)
%
M = M1*M2;
dx = L1 / M1;
dy = L2 / M2;
x = 0:dx:L1-dx;
y = 0:dy:L2-dy;

%% time step size
T = 0.1; % time domain = [0,T]
dt = 0.01;
t = 0:dt:T;

%% difference matrices
FDx=1/dx*spdiags([ones(M1,1),-ones(M1,1),ones(M1,1)],[-M1+1,0,1],M1,M1); % forward difference
FDy=1/dy*spdiags([ones(M2,1),-ones(M2,1),ones(M2,1)],[-M2+1,0,1],M2,M2); % backward difference
D2x=1/(dx^2)*spdiags([ones(M1,1),ones(M1,1),-2*ones(M1,1),ones(M1,1),ones(M1,1)],[-M1+1,-1,0,1,M1-1],M1,M1); % 2nd order central difference for x-variable
D2y=1/(dy^2)*spdiags([ones(M2,1),ones(M2,1),-2*ones(M2,1),ones(M2,1),ones(M2,1)],[-M2+1,-1,0,1,M2-1],M2,M2); % 2nd order central difference for y-variable

%% matrix A (corresponding to Laplacian)
A = -(kron(eye(M2),D2x)+kron(D2y,eye(M1)));

%% matrix for numerical solution
U = zeros(M1,M2);
%Phi = zeros(M1,M2); % for absolute values

%% Norm and Energy
Norm = zeros(1,length(t));
Energy =  zeros(1,length(t));

%% Initial condition
U = ones(M1,M2) + 2*repmat(cos(x'),[1,M2])+2*repmat(cos(y'),[1,M1])';
%Phi(:,1) = abs(U(:,1));

%% Norm and Energy at t = 0
Norm(1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
Energy(1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);

%% numerical methods
%type = 'MB4'; %EXACT, RK4, GAUSS2, GAUSS4, AVF2, AVF4, MB4

tic
switch type
    %% 4-stage 4th order explicit Runge kutta method
    case 'RK4   ' % large dt may cause instability
        [Norm, Energy] = rk4(dx, dy, L1, L2, FDx, FDy, D2x, D2y, dt, t, U, Norm, Energy);

    %% 2nd order Gauss method (symplectic)
    case 'GAUSS2'
        [Norm, Energy] = gauss2(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy);

    %% 4th order Gauss method (symplectic)
    case 'GAUSS4'
        [Norm, Energy] = gauss4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy);

    %% 2nd order AVF method
    case 'AVF2  ' % 2次 AVF method
        [Norm, Energy] = avf2(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy);

    %% 4nd order AVF collocation method
    case 'AVF4  '
        [Norm, Energy] = avf4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy);

    %% 4th order parallel method (MB4)
    case 'MB4   '
        [Norm, Energy] = mb4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, quadtol, dt, t, U, Norm, Energy);
    otherwise
        error('unknown method')
end
time = toc;

ErrorEnergy = abs(Energy(1)*ones(1,length(t))-Energy);
ErrorNorm = abs(Norm(1)*ones(1,length(t))-Norm);
save('ErrorEnergy.txt', 'ErrorEnergy')
save('ErrorNorm.txt', 'ErrorNorm')

% figure
% hold on
% plot(t,Energy)
% plot(t,Norm)

%figure
%semilogy(t,ErrorEnergy,'-b','linewidth',2)
%hold on
%semilogy(t,ErrorNorm,'-r','linewidth',2)
%legend('Energy','Norm')
%xlabel('time','FontSize',16)
%ylabel('error','FontSize',16)
%

% mesh(y,x,U)
end

%function [Norm, Energy] = exact(dx, dy, L1, L2, FDx, FDy, D2x, D2y, dt, t, U, Norm, Energy)
%[tode45, uode45] = ode45(@(tt,uu) sqrt(-1)*(D2x*uu+uu*D2y) + sqrt(-1)*abs(uu).^2.*uu, t, U, odeset('reltol',ode45tol,'abstol',ode45tol));
%for iter = 1 : size(uode45, )
%Norm(:) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
%Energy(:) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
%end

function [Norm, Energy] = rk4(dx, dy, L1, L2, FDx, FDy, D2x, D2y, dt, t, U, Norm, Energy)
func = @(u) sqrt(-1)*(D2x*u+u*D2y) + sqrt(-1)*abs(u).^2.*u;
for iter = 1:length(t)-1
    iter;
    U;
    k1 = func(U);
    k2 = func(U+dt/2*k1);
    k3 = func(U+dt/2*k2);
    k4 = func(U+dt*k3);
    U = U + dt/6*(k1+2*k2+2*k3+k4);
    %save(sprintf("U_RK4_%03d.txt", iter), "U")
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
end
end

function [Norm, Energy] = gauss2(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy)
a=1/2;% RK coefficient
P=reshape(real(U),[M,1]); %real(U)
Q=reshape(imag(U),[M,1]); %imag(U)

for iter = 1:length(t)-1
    G2func = @(P1,Q1,p0,q0) [P1-p0;Q1-q0] - dt*a*[A*Q1 + eps*(P1.^2+Q1.^2).*Q1; -A*P1 - eps*(P1.^2+Q1.^2).*P1]; % p1 and q1 satisfying G2func=0
    p0 = P;
    q0 = Q;
    Jf = [2*eps*diag(p0.*q0), A + eps * diag(p0.^2+3*q0.^2); -A - eps * diag(p0.^2+3*q0.^2), 2*eps*diag(p0.*q0)]; % Jacobian of the vector field
    [LUl, LUu] = lu(eye(2*M)-dt*a*Jf); % LU decomposition for the approximated Jacobian for G2func
    sniter = 0; % counter for simplified Newton method
    rho = ones(2*M,1);
    P1 = p0; Q1 = q0; % initial guess (could be improved)
    while (max(abs(rho)) > TolX) % simplified Newton iteration (max iteration: 20)
        sniter = sniter +1;
        rho = - LUu \ (LUl \ G2func(P1,Q1,p0,q0));
        P1 = P1 + rho(1:M1*M2,1); Q1 = Q1 + rho(M1*M2+1:2*M1*M2,1);
        norm(rho);
        if sniter==20, break, end
    end
    P =2*P1-P;
    Q =2*Q1-Q;
    U = reshape(P + sqrt(-1)*Q,[M1,M2]);
    %save(sprintf("U_GAUSS2_%03d.txt", iter), "U")
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
end
end

function [Norm, Energy] = gauss4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy)
RKA = [1/4, 1/4 - sqrt(3)/6; 1/4 + sqrt(3)/6, 1/4];% RK coefficients
P=reshape(real(U),[M1*M2,1]); %real(U)
Q=reshape(imag(U),[M1*M2,1]); %imag(U)

for iter = 1:length(t)-1
    G4func = @(P1,Q1,P2,Q2,p0,q0) [P1-p0;Q1-q0;P2-p0;Q2-q0] - dt*kron(RKA,eye(2*M))*[A*Q1 + eps*(P1.^2+Q1.^2).*Q1; -A*P1 - eps*(P1.^2+Q1.^2).*P1; A*Q2 + eps* (P2.^2+Q2.^2).*Q2; -A*P2 - eps* (P2.^2+Q2.^2).*P2];
    % P1,Q1,P2,Q2 satisfying G4func=0
    p0 = P; q0 = Q;
    Jf = [2*eps*diag(p0.*q0), A + eps * diag(p0.^2+3*q0.^2); -A - eps * diag(p0.^2+3*q0.^2), 2*eps*diag(p0.*q0)]; % Jacobian of the vector field
    P1 = p0; Q1 = q0; P2 = p0; Q2 = q0; % initial guess (could be improved)
    [LUl,LUu] = lu(eye(4*M)-dt*kron(RKA,Jf)); % LU decomposition for the approximated Jacobian for G4func
    sniter = 0; % counter for simplified Newton method
    rho = ones(4*M,1);
    while (max(abs(rho)) > TolX) % simplified Newton iteration (max iteration: 20)
        sniter = sniter +1;
        rho = - LUu \ (LUl \ G4func(P1,Q1,P2,Q2,p0,q0));
        P1 = P1 + rho(1:M,1); Q1 = Q1 + rho(M+1:2*M,1);
        P2 = P2 + rho(2*M+1:3*M,1); Q2 = Q2 + rho(3*M+1:4*M,1);
        norm(rho);
        if sniter==20, break, end
    end
    P = P + sqrt(3)*(P2-P1);
    Q = Q + sqrt(3)*(Q2-Q1);
    U = reshape(P + sqrt(-1)*Q,[M1,M2]);
    %save(sprintf("U_GAUSS4_%03d.txt", iter), "U")
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
end
end

function [Norm, Energy] = avf2(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy)
E=1/2; % CSRK coefficient
P=reshape(real(U),[M1*M2,1]); %real(U)
Q=reshape(imag(U),[M1*M2,1]); %imag(U)

for iter = 1:length(t)-1
    AVFfunc = @(P1,Q1,p0,q0) [P1-p0;Q1-q0] - dt*vertcat(A*(Q1+q0)/2 + eps* (3*q0.^3+3*Q1.*(q0.^2)+3*(Q1.^2).*q0+3*(p0.^2).*q0+2*P1.*p0.*q0+(P1.^2).*q0+3*Q1.^3+(p0.^2).*Q1+2*P1.*p0.*Q1+3*(P1.^2).*Q1)/12,...
        -A*(P1+p0)/2 -eps* ((3*p0+P1).*(q0.^2)+(2*p0+2*P1).*Q1.*q0+(p0+3*P1).*(Q1.^2)+3*(p0.^3)+3*P1.*(p0.^2)+3*(P1.^2).*p0+3*(P1.^3))/12);% P1 and Q1 satisfying G4func=0
    p0 = P;
    q0 = Q;
    Jf = [2*eps*diag(p0.*q0), A + eps * diag(p0.^2+3*q0.^2); -A - eps * diag(p0.^2+3*q0.^2), 2*eps*diag(p0.*q0)]; % Jacobian of the vector field
    P1 = p0;
    Q1 = q0; % initial guess (could be improved)
    [LUl,LUu] = lu(eye(2*M)-dt*E*Jf); % LU decomposition for the approximated Jacobian for AVFfunc
    sniter = 0;
    rho = ones(2*M,1);
    while (max(abs(rho)) > TolX) % simplified Newton iteration (max iteration: 20)
        sniter = sniter +1;
        rho = - LUu \ (LUl \ AVFfunc(P1,Q1,p0,q0));
        P1 = P1 + rho(1:M,1); Q1 = Q1 + rho(M+1:2*M,1);
        norm(rho);
        if sniter==20, break, end
    end
    P = P1;
    Q = Q1;
    U = reshape(P + sqrt(-1)*Q,[M1,M2]);
    %save(sprintf("U_AVF2_%03d.txt", iter), "U")
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
end
end

function [Norm, Energy] = avf4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, dt, t, U, Norm, Energy)
E = [1/3, - 1/24; 2/3, 1/6]; % CSRK coefficient
P=reshape(real(U),[M1*M2,1]); %real(U)
Q=reshape(imag(U),[M1*M2,1]); %imag(U)

for iter = 1:length(t)-1
    AVF4func = @(P1,Q1,P2,Q2,p0,q0) [P1-p0;Q1-q0;P2-p0;Q2-q0] - dt*[A*(350.*q0-70.*Q2+560.*Q1)/1680+eps*((177.*q0.^3+(228.*Q1-45.*Q2).*q0.^2+ ...
        (9.*Q2.^2-96.*Q1.*Q2+240.*Q1.^2+177.*p0.^2+(152.*P1-30.*P2).*p0+3.*P2.^2-32.*P1.*P2+80.*P1.^2).*q0-21.*Q2.^3+12.*Q1.*Q2.^2+ ...
        (-48.*Q1.^2-15.*p0.^2+(6.*P2-32.*P1).*p0-21.*P2.^2+8.*P1.*P2-16.*P1.^2).*Q2+384.*Q1.^3+(76.*p0.^2+(160.*P1-32.*P2).*p0+4.*P2.^2-32.*P1.*P2+384.*P1.^2).*Q1)/1680); ...
        -A*(350.*p0-70.*P2+560.*P1)/1680-eps*((177.*p0-15.*P2+76.*P1).*q0.^2+((-30.*p0+6.*P2-32.*P1).*Q2+(152.*p0-32.*P2+160.*P1).*Q1).*q0+ ...
        (3.*p0-21.*P2+4.*P1).*Q2.^2+(-32.*p0+8.*P2-32.*P1).*Q1.*Q2+(80.*p0-16.*P2+384.*P1).*Q1.^2+177.*p0.^3+(228.*P1-45.*P2).*p0.^2+(9.*P2.^2-96.*P1.*P2+240.*P1.^2).*p0- ...
        21.*P2.^3+12.*P1.*P2.^2-48.*P1.^2.*P2+384.*P1.^3)/1680; ...
        A*(70.*q0+70.*Q2+280.*Q1)/420+eps*((39.*q0.^3+(60.*Q1-9.*Q2).*q0.^2+(-9.*Q2.^2-48.*Q1.*Q2+48.*Q1.^2+39.*p0.^2+(40.*P1-6.*P2).*p0-3.*P2.^2-16.*P1.*P2+16.*P1.^2).*q0 ...
        +39.*Q2.^3+60.*Q1.*Q2.^2+(48.*Q1.^2-3.*p0.^2+(-6.*P2-16.*P1).*p0+39.*P2.^2+40.*P1.*P2+16.*P1.^2).*Q2+192.*Q1.^3+ ...
        (20.*p0.^2+(32.*P1-16.*P2).*p0+20.*P2.^2+32.*P1.*P2+192.*P1.^2).*Q1)/420);...
        -A*(70.*p0+70.*P2+280.*P1)/420-eps*((39.*p0-3.*P2+20.*P1).*q0.^2+((-6.*p0-6.*P2-16.*P1).*Q2+(40.*p0-16.*P2+32.*P1).*Q1).*q0+(-3.*p0+39.*P2+20.*P1).*Q2.^2 ...
        +(-16.*p0+40.*P2+32.*P1).*Q1.*Q2+(16.*p0+16.*P2+192.*P1).*Q1.^2+39.*p0.^3+(60.*P1-9.*P2).*p0.^2+(-9.*P2.^2-48.*P1.*P2+48.*P1.^2).*p0+39.*P2.^3+60.*P1.*P2.^2+48.*P1.^2.* ...
        P2+192.*P1.^3)/420]; % P1,Q1,P2,Q2 satisfying AVF4func=0
    p0 = P; q0 = Q;
    Jf = [2*eps*diag(p0.*q0), A + eps * diag(p0.^2+3*q0.^2); -A - eps * diag(p0.^2+3*q0.^2), 2*eps*diag(p0.*q0)]; % Jacobian of the vector field
    P1 = p0; Q1 = q0; P2 = p0; Q2 = q0; % initial guess (could be improved)
    [LUl,LUu] = lu(eye(4*M)-dt*kron(E,Jf)); % LU decomposition for the approximated Jacobian for AVF4func
    sniter = 0;
    rho = ones(4*M,1);
    while (max(abs(rho)) > TolX) % simplified Newton iteration (max iteration: 20)
        sniter = sniter +1;
        rho = - LUu \ (LUl \ AVF4func(P1,Q1,P2,Q2,p0,q0));
        P1 = P1 + rho(1:M,1); Q1 = Q1 + rho(M+1:2*M,1);
        P2 = P2 + rho(2*M+1:3*M,1); Q2 = Q2 + rho(3*M+1:4*M,1);
        norm(rho);
        if sniter==20, break, end
    end
    P = P2; Q = Q2;
    U = reshape(P + sqrt(-1)*Q,[M1,M2]);
    %save(sprintf("U_AVF4_%03d.txt", iter), "U")
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);
end
end

function [Norm, Energy] = mb4(dx, dy, M1, M2, M, L1, L2, FDx, FDy, A, eps, TolX, quadtol, dt, t, U, Norm, Energy)
P=reshape(real(U),[M1*M2,1]); %real(U)
Q=reshape(imag(U),[M1*M2,1]); %imag(U)

%% preparation for parallel computation
theta = 19/8;
[E,Tmat,Lam] = gen_matE(theta,1/3,2/3,1,quadtol); % Tinv * E * T = Lam
Tinv = inv(Tmat);
lambda1 = Lam(1,1); lambda2 = Lam(2,2); lambda3 = Lam(3,3); % eigenvalues of E
kronTinv=kron(Tinv,eye(2*M));
kronT=kron(Tmat,eye(2*M));

for iter = 1:length(t)-1
    MB4func = @(P1,Q1,P2,Q2,P3,Q3,p0,q0) [P1-p0;Q1-q0;P2-p0;Q2-q0;P3-p0;Q3-q0] - dt*[(-A*((1108800.*q0+1108800.*Q3-1108800.*Q2-1108800.*Q1).*theta-68376.*q0+12936.*Q3+16632.*Q2-182952.*Q1)-eps*((582600.*q0.^3+ ...
        (70800.*Q3-307800.*Q2+680400.*Q1).*q0.^2+(70800.*Q3.^2+(43200.*Q2+43200.*Q1).*Q3+243000.*Q2.^2-388800.*Q1.*Q2+680400.*Q1.^2+582600.*p0.^2+ ...
        (47200.*P3-205200.*P2+453600.*P1).*p0+23600.*P3.^2+(14400.*P2+14400.*P1).*P3+81000.*P2.^2-129600.*P1.*P2+226800.*P1.^2).*q0+582600.*Q3.^3+ ...
        (680400.*Q2-307800.*Q1).*Q3.^2+(680400.*Q2.^2-388800.*Q1.*Q2+243000.*Q1.^2+23600.*p0.^2+(47200.*P3+14400.*P2+14400.*P1).*p0+582600.*P3.^2+ ...
        (453600.*P2-205200.*P1).*P3+226800.*P2.^2-129600.*P1.*P2+81000.*P1.^2).*Q3-729000.*Q2.^3-874800.*Q1.*Q2.^2+(-874800.*Q1.^2-102600.*p0.^2+ ...
        (14400.*P3+162000.*P2-129600.*P1).*p0+226800.*P3.^2+(453600.*P2-129600.*P1).*P3-729000.*P2.^2-583200.*P1.*P2-291600.*P1.^2).*Q2-729000.*Q1.^3+ ...
        (226800.*p0.^2+(14400.*P3-129600.*P2+453600.*P1).*p0-102600.*P3.^2+(162000.*P1-129600.*P2).*P3-291600.*P2.^2-583200.*P1.*P2-729000.*P1.^2).*Q1).*theta- ...
        33111.*q0.^3+(-4716.*Q3+22761.*Q2-55728.*Q1).*q0.^2+(756.*Q3.^2+(6156.*Q2-13284.*Q1).*Q3-6561.*Q2.^2+61236.*Q1.*Q2-78732.*Q1.^2-33111.*p0.^2+ ...
        (-3144.*P3+15174.*P2-37152.*P1).*p0+252.*P3.^2+(2052.*P2-4428.*P1).*P3-2187.*P2.^2+20412.*P1.*P2-26244.*P1.^2).*q0+9549.*Q3.^3+(12960.*Q2-6723.*Q1).* ...
        Q3.^2+ ...
        (14580.*Q2.^2+2916.*Q1.*Q2-9477.*Q1.^2-1572.*p0.^2+(504.*P3+2052.*P2-4428.*P1).*p0+9549.*P3.^2+(8640.*P2-4482.*P1).*P3+4860.*P2.^2+972.*P1.*P2-3159.*P1.^2).* ...
        Q3-6561.*Q2.^3-52488.*Q1.*Q2.^2+ ...
        (52488.*Q1.^2+7587.*p0.^2+(2052.*P3-4374.*P2+20412.*P1).*p0+4320.*P3.^2+(9720.*P2+972.*P1).*P3-6561.*P2.^2-34992.*P1.*P2+17496.*P1.^2).*Q2-137781.*Q1.^3+ ...
        (-18576.*p0.^2+(-4428.*P3+20412.*P2-52488.*P1).*p0-2241.*P3.^2+(972.*P2-6318.*P1).*P3-17496.*P2.^2+34992.*P1.*P2-137781.*P1.^2).*Q1))/665280; ...
        -(-A*((1108800.*p0+1108800.*P3-1108800.*P2-1108800.*P1).*theta-68376.*p0+12936.*P3+16632.*P2-182952.*P1)-eps*(( ...
        (582600.*p0+23600.*P3-102600.*P2+226800.*P1).*q0.^2+((47200.*p0+47200.*P3+14400.*P2+14400.*P1).*Q3+ ...
        (-205200.*p0+14400.*P3+162000.*P2-129600.*P1).*Q2+(453600.*p0+14400.*P3-129600.*P2+453600.*P1).*Q1).*q0+ ...
        (23600.*p0+582600.*P3+226800.*P2-102600.*P1).*Q3.^2+ ...
        ((14400.*p0+453600.*P3+453600.*P2-129600.*P1).*Q2+(14400.*p0-205200.*P3-129600.*P2+162000.*P1).*Q1).*Q3+ ...
        (81000.*p0+226800.*P3-729000.*P2-291600.*P1).*Q2.^2+(-129600.*p0-129600.*P3-583200.*P2-583200.*P1).*Q1.*Q2+ ...
        (226800.*p0+81000.*P3-291600.*P2-729000.*P1).*Q1.^2+582600.*p0.^3+(70800.*P3-307800.*P2+680400.*P1).*p0.^2+ ...
        (70800.*P3.^2+(43200.*P2+43200.*P1).*P3+243000.*P2.^2-388800.*P1.*P2+680400.*P1.^2).*p0+582600.*P3.^3+(680400.*P2-307800.*P1).*P3.^2+ ...
        (680400.*P2.^2-388800.*P1.*P2+243000.*P1.^2).*P3-729000.*P2.^3-874800.*P1.*P2.^2-874800.*P1.^2.*P2-729000.*P1.^3).*theta+ ...
        (-33111.*p0-1572.*P3+7587.*P2-18576.*P1).*q0.^2+ ...
        ((-3144.*p0+504.*P3+2052.*P2-4428.*P1).*Q3+(15174.*p0+2052.*P3-4374.*P2+20412.*P1).*Q2+(-37152.*p0-4428.*P3+20412.*P2-52488.*P1).*Q1).*q0+ ...
        (252.*p0+9549.*P3+4320.*P2-2241.*P1).*Q3.^2+((2052.*p0+8640.*P3+9720.*P2+972.*P1).*Q2+(-4428.*p0-4482.*P3+972.*P2-6318.*P1).*Q1).*Q3+ ...
        (-2187.*p0+4860.*P3-6561.*P2-17496.*P1).*Q2.^2+(20412.*p0+972.*P3-34992.*P2+34992.*P1).*Q1.*Q2+(-26244.*p0-3159.*P3+17496.*P2-137781.*P1).*Q1.^2- ...
        33111.*p0.^3+(-4716.*P3+22761.*P2-55728.*P1).*p0.^2+(756.*P3.^2+(6156.*P2-13284.*P1).*P3-6561.*P2.^2+61236.*P1.*P2-78732.*P1.^2).*p0+9549.*P3.^3+ ...
        (12960.*P2-6723.*P1).*P3.^2+(14580.*P2.^2+2916.*P1.*P2-9477.*P1.^2).*P3-6561.*P2.^3-52488.*P1.*P2.^2+52488.*P1.^2.*P2-137781.*P1.^3))/665280; ...
        -(-A*((277200.*q0+277200.*Q3-277200.*Q2-277200.*Q1).*theta+24024.*q0+3696.*Q3+16632.*Q2+66528.*Q1)-eps*((145650.*q0.^3+ ...
        (17700.*Q3-76950.*Q2+170100.*Q1).*q0.^2+(17700.*Q3.^2+(10800.*Q2+10800.*Q1).*Q3+60750.*Q2.^2-97200.*Q1.*Q2+170100.*Q1.^2+145650.*p0.^2+ ...
        (11800.*P3-51300.*P2+113400.*P1).*p0+5900.*P3.^2+(3600.*P2+3600.*P1).*P3+20250.*P2.^2-32400.*P1.*P2+56700.*P1.^2).*q0+145650.*Q3.^3+ ...
        (170100.*Q2-76950.*Q1).*Q3.^2+(170100.*Q2.^2-97200.*Q1.*Q2+60750.*Q1.^2+5900.*p0.^2+(11800.*P3+3600.*P2+3600.*P1).*p0+145650.*P3.^2+ ...
        (113400.*P2-51300.*P1).*P3+56700.*P2.^2-32400.*P1.*P2+20250.*P1.^2).*Q3-182250.*Q2.^3-218700.*Q1.*Q2.^2+ ...
        (-218700.*Q1.^2-25650.*p0.^2+(3600.*P3+40500.*P2-32400.*P1).*p0+56700.*P3.^2+(113400.*P2-32400.*P1).*P3-182250.*P2.^2-145800.*P1.*P2-72900.*P1.^2).*Q2- ...
        182250.*Q1.^3+(56700.*p0.^2+(3600.*P3-32400.*P2+113400.*P1).*p0-25650.*P3.^2+(40500.*P1-32400.*P2).*P3-72900.*P2.^2-145800.*P1.*P2-182250.*P1.^2).*Q1).*theta ...
        +11223.*q0.^3+(1674.*Q3-7695.*Q2+19278.*Q1).*q0.^2+(306.*Q3.^2+(4212.*Q1-648.*Q2).*Q3+3645.*Q2.^2-23328.*Q1.*Q2+27702.*Q1.^2+11223.*p0.^2+ ...
        (1116.*P3-5130.*P2+12852.*P1).*p0+102.*P3.^2+(1404.*P1-216.*P2).*P3+1215.*P2.^2-7776.*P1.*P2+9234.*P1.^2).*q0+558.*Q3.^3+(2106.*Q2-324.*Q1).*Q3.^2+ ...
        (4374.*Q2.^2-8748.*Q1.*Q2+4374.*Q1.^2+558.*p0.^2+(204.*P3-216.*P2+1404.*P1).*p0+558.*P3.^2+(1404.*P2-216.*P1).*P3+1458.*P2.^2-2916.*P1.*P2+1458.*P1.^2).*Q3+ ...
        19683.*Q2.^3+13122.*Q1.*Q2.^2+ ...
        (-13122.*Q1.^2-2565.*p0.^2+(-216.*P3+2430.*P2-7776.*P1).*p0+702.*P3.^2+(2916.*P2-2916.*P1).*P3+19683.*P2.^2+8748.*P1.*P2-4374.*P1.^2).*Q2+52488.*Q1.^3+ ...
        (6426.*p0.^2+(1404.*P3-7776.*P2+18468.*P1).*p0-108.*P3.^2+(2916.*P1-2916.*P2).*P3+4374.*P2.^2-8748.*P1.*P2+52488.*P1.^2).*Q1))/166320; ...
        (-A*((277200.*p0+277200.*P3-277200.*P2-277200.*P1).*theta+24024.*p0+3696.*P3+16632.*P2+66528.*P1)-eps*(( ...
        (145650.*p0+5900.*P3-25650.*P2+56700.*P1).*q0.^2+ ...
        ((11800.*p0+11800.*P3+3600.*P2+3600.*P1).*Q3+(-51300.*p0+3600.*P3+40500.*P2-32400.*P1).*Q2+(113400.*p0+3600.*P3-32400.*P2+113400.*P1).*Q1).*q0 ...
        +(5900.*p0+145650.*P3+56700.*P2-25650.*P1).*Q3.^2+((3600.*p0+113400.*P3+113400.*P2-32400.*P1).*Q2+(3600.*p0-51300.*P3-32400.*P2+40500.*P1).*Q1).* ...
        Q3+(20250.*p0+56700.*P3-182250.*P2-72900.*P1).*Q2.^2+(-32400.*p0-32400.*P3-145800.*P2-145800.*P1).*Q1.*Q2+ ...
        (56700.*p0+20250.*P3-72900.*P2-182250.*P1).*Q1.^2+145650.*p0.^3+(17700.*P3-76950.*P2+170100.*P1).*p0.^2+ ...
        (17700.*P3.^2+(10800.*P2+10800.*P1).*P3+60750.*P2.^2-97200.*P1.*P2+170100.*P1.^2).*p0+145650.*P3.^3+(170100.*P2-76950.*P1).*P3.^2+ ...
        (170100.*P2.^2-97200.*P1.*P2+60750.*P1.^2).*P3-182250.*P2.^3-218700.*P1.*P2.^2-218700.*P1.^2.*P2-182250.*P1.^3).*theta+(11223.*p0+558.*P3-2565.*P2+6426.*P1).* ...
        q0.^2+((1116.*p0+204.*P3-216.*P2+1404.*P1).*Q3+(-5130.*p0-216.*P3+2430.*P2-7776.*P1).*Q2+(12852.*p0+1404.*P3-7776.*P2+18468.*P1).*Q1).*q0+ ...
        (102.*p0+558.*P3+702.*P2-108.*P1).*Q3.^2+((-216.*p0+1404.*P3+2916.*P2-2916.*P1).*Q2+(1404.*p0-216.*P3-2916.*P2+2916.*P1).*Q1).*Q3+ ...
        (1215.*p0+1458.*P3+19683.*P2+4374.*P1).*Q2.^2+(-7776.*p0-2916.*P3+8748.*P2-8748.*P1).*Q1.*Q2+(9234.*p0+1458.*P3-4374.*P2+52488.*P1).*Q1.^2+11223.* ...
        p0.^3+(1674.*P3-7695.*P2+19278.*P1).*p0.^2+(306.*P3.^2+(4212.*P1-648.*P2).*P3+3645.*P2.^2-23328.*P1.*P2+27702.*P1.^2).*p0+558.*P3.^3+(2106.*P2-324.*P1).*P3.^2 ...
        +(4374.*P2.^2-8748.*P1.*P2+4374.*P1.^2).*P3+19683.*P2.^3+13122.*P1.*P2.^2-13122.*P1.^2.*P2+52488.*P1.^3))/166320; ...
        -(-A*(840.*q0+840.*Q3+2520.*Q2+2520.*Q1)-eps*(357.*q0.^3+(60.*Q3-243.*Q2+648.*Q1).*q0.^2+(60.*Q3.^2+(108.*Q2+108.*Q1).*Q3+243.*Q2.^2-972.*Q1.*Q2+ ...
        972.*Q1.^2+357.*p0.^2+(40.*P3-162.*P2+432.*P1).*p0+20.*P3.^2+(36.*P2+36.*P1).*P3+81.*P2.^2-324.*P1.*P2+324.*P1.^2).*q0+357.*Q3.^3+(648.*Q2-243.*Q1).*Q3.^2+ ...
        (972.*Q2.^2-972.*Q1.*Q2+243.*Q1.^2+20.*p0.^2+(40.*P3+36.*P2+36.*P1).*p0+357.*P3.^2+(432.*P2-162.*P1).*P3+324.*P2.^2-324.*P1.*P2+81.*P1.^2).*Q3+2187.*Q2.^3+ ...
        (-81.*p0.^2+(36.*P3+162.*P2-324.*P1).*p0+216.*P3.^2+(648.*P2-324.*P1).*P3+2187.*P2.^2).*Q2+2187.*Q1.^3+ ...
        (216.*p0.^2+(36.*P3-324.*P2+648.*P1).*p0-81.*P3.^2+(162.*P1-324.*P2).*P3+2187.*P1.^2).*Q1))/6720; ...
        (-A*(840.*p0+840.*P3+2520.*P2+2520.*P1)-eps*((357.*p0+20.*P3-81.*P2+216.*P1).*q0.^2+ ...
        ((40.*p0+40.*P3+36.*P2+36.*P1).*Q3+(-162.*p0+36.*P3+162.*P2-324.*P1).*Q2+(432.*p0+36.*P3-324.*P2+648.*P1).*Q1).*q0+(20.*p0+357.*P3+216.*P2-81.*P1).* ...
        Q3.^2+((36.*p0+432.*P3+648.*P2-324.*P1).*Q2+(36.*p0-162.*P3-324.*P2+162.*P1).*Q1).*Q3+(81.*p0+324.*P3+2187.*P2).*Q2.^2+(-324.*p0-324.*P3).*Q1.*Q2+ ...
        (324.*p0+81.*P3+2187.*P1).*Q1.^2+357.*p0.^3+(60.*P3-243.*P2+648.*P1).*p0.^2+(60.*P3.^2+(108.*P2+108.*P1).*P3+243.*P2.^2-972.*P1.*P2+972.*P1.^2).*p0+357.*P3.^3+ ...
        (648.*P2-243.*P1).*P3.^2+(972.*P2.^2-972.*P1.*P2+243.*P1.^2).*P3+2187.*P2.^3+2187.*P1.^3))/6720];
        % P1,Q1,P2,Q2,P3,Q3 satisfying MB4func=0 (numerical integration may be more efficient)

    p0 = P; q0 = Q;
    Jf = [2*eps*diag(p0.*q0), A + eps * diag(p0.^2+3*q0.^2); -A - eps * diag(p0.^2+3*q0.^2), 2*eps*diag(p0.*q0)]; % Jacobian of the vector field
    P1 = p0; Q1 = q0; P2 = p0; Q2 = q0; P3 = p0; Q3 = q0; % initial guess (could be improved)

    [LUl1,LUu1] = lu(eye(2*M)-dt*lambda1*Jf); [LUl2,LUu2] = lu(eye(2*M)-dt*lambda2*Jf); [LUl3,LUu3] = lu(eye(2*M)-dt*lambda3*Jf); % LU decomposition for the approximated Jacobian for MB4func
    sniter = 0;
    rho = ones(6*M,1);
    while (max(abs(rho)) > TolX) % simplified Newton iteration (max iteration: 20)
        sniter = sniter +1;
        modMB4 = kronTinv*MB4func(P1,Q1,P2,Q2,P3,Q3,p0,q0);

        rho1 = - LUu1 \ (LUl1 \ modMB4(1:2*M,1)); % these 3 lines can be computed in parallel
        rho2 = - LUu2 \ (LUl2 \ modMB4(2*M+1:4*M,1));
        rho3 = - LUu3 \ (LUl3 \ modMB4(4*M+1:6*M,1));

        rho = kronT*[rho1;rho2;rho3];

        P1 = P1 + rho(1:M,1); Q1 = Q1 + rho(M+1:2*M,1);
        P2 = P2 + rho(2*M+1:3*M,1); Q2 = Q2 + rho(3*M+1:4*M,1);
        P3 = P3 + rho(4*M+1:5*M,1); Q3 = Q3 + rho(5*M+1:6*M,1);
        norm(rho);
        if sniter==20, break, end
    end


    P = P3; Q = Q3;
    U = reshape(P + sqrt(-1)*Q,[M1,M2]);
    %U_for_save = cat(2, P, Q);
    %save(sprintf('MB4_U_%03d.txt', iter), 'U_for_save', '-ascii','-double');
    Norm(iter+1) = sum(sum(abs(U).^2))*dx*dy/(L1*L2);
    Energy(iter+1) = sum(sum(abs(FDx*U).^2 + abs(U*FDy).^2 + eps/2*abs(U).^4))*dx*dy/(L1*L2);

end
end

function [C,T,L] = gen_matE(theta , c1,c2,c3 ,quadtol)

l1 = @(s) s./c1.*(s-c2)./(c1-c2).*(s-c3)./(c1-c3);
l2 = @(s) s./c2.*(s-c1)./(c2-c1).*(s-c3)./(c2-c3);
l3 = @(s) s./c3.*(s-c1)./(c3-c1).*(s-c2)./(c3-c2);

eta = 300*theta-1;
a = -12-12*eta;
b = 6 + 6*eta;
c = -2 - 2*eta;
d = -12 - 18*eta;
e = 3*eta;
f = 3-eta;

A = @(t,s) a.*t.^3.*s.^2 + 2.*b.*t.^3.*s + c.*t.^3 + 3.*b.*t.^2.*s.^2 + d.*t.^2.*s + e.*t.^2 ...
    + 3.*c.*t.*s.^2 + 2.*e.*t.*s + f.*t;

c11 = @(s) A(c1,s).*l1(s);
c12 = @(s) A(c1,s).*l2(s);
c13 = @(s) A(c1,s).*l3(s);

c21 = @(s) A(c2,s).*l1(s);
c22 = @(s) A(c2,s).*l2(s);
c23 = @(s) A(c2,s).*l3(s);

c31 = @(s) A(c3,s).*l1(s);
c32 = @(s) A(c3,s).*l2(s);
c33 = @(s) A(c3,s).*l3(s);

C = [integral(c11,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c12,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c13,0,1,'AbsTol',quadtol,'RelTol',quadtol);...
    integral(c21,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c22,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c23,0,1,'AbsTol',quadtol,'RelTol',quadtol);...
    integral(c31,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c32,0,1,'AbsTol',quadtol,'RelTol',quadtol),integral(c33,0,1,'AbsTol',quadtol,'RelTol',quadtol)];


[T,L] = eig(C);
end
