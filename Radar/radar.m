function [R] = radar(X, A, L, alpha, beta, gamma, niters)

    [n,d] = size(X);
    Dr = eye(n);
    Dw = eye(n);
    R = inv(eye(n)+beta*Dr+gamma*L)*X;
    for iter = 1:niters
        %% update W
        W = inv(X*X'+alpha*Dw)*(X*X'-X*R');
        Wtmp = sqrt(sum(W.*W,2)+eps);
        Dw = diag(0.5./Wtmp);
        
        %% update R
        R = inv(eye(n)+beta*Dr+gamma*L)*(X-W'*X);
        Rtmp = sqrt(sum(R.*R,2))+eps;
        Dr = diag(0.5./Rtmp);
        
        %% check if the objective function converges
        obj(iter) = norm(X-W'*X-R,'fro')^2 + alpha*sum(Wtmp) + beta*sum(Rtmp) + gamma*trace(R'*L*R);
        fprintf('the object value in iter %d is %f\n', iter, obj(iter));
        if (iter >= 2 && abs(obj(iter)-obj(iter-1))<1e-3)
            break;
        end
    end
end